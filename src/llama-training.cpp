#include "llama-training.h"
#include "llama-adapter.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

struct llama_adapter_training {
    llama_model * model;
    llama_adapter_lora * adapter;
    llama_adapter_training_params config;
    ggml_backend_ptr cpu;
    llama_adapter_cvec cvec;
    llama_adapter_loras loras;
    std::unique_ptr<llm_graph_result> result;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * labels = nullptr;
    ggml_tensor * loss = nullptr;
    ggml_tensor * upstream_tensor = nullptr;
    ggml_tensor * input_gradient = nullptr;
    ggml_cgraph * forward = nullptr;
    ggml_context_ptr forward_context;
    ggml_cplan plan = {};
    std::vector<uint8_t> scratch;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    llm_graph_params graph_params = {};
    std::map<std::string, ggml_tensor *> gradients;
    size_t workspace = 0;
    bool valid = false;
    bool consumed = false;
    uint64_t supervised = 0;
    uint64_t revision = 0;
    std::string error;
    bool checkpointed = false;
    llama_adapter_checkpoint_store store = {};
    ggml_context_ptr gradient_context;
    std::map<std::string, std::vector<float>> gradient_values;
    std::vector<float> boundary, upstream;
    std::vector<std::vector<float>> boundaries;
    size_t retained_workspace = 0;
};

static float compute_checkpointed(llama_adapter_training * t, const llama_token * tokens, const llama_token * labels,
        ggml_abort_callback abort_callback, void * abort_data);

static size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) throw std::runtime_error("training workspace size overflow");
    return a + b;
}

static void validate_training_model(llama_model * model, llama_adapter_lora * adapter, llama_adapter_training_params params) {
    if (!model || !adapter || adapter->model != model || adapter->ab_map.empty() || !adapter->alora_invocation_tokens.empty() ||
            !std::isfinite(adapter->alpha) || adapter->alpha <= 0) {
        throw std::runtime_error("training requires a matching initialized LoRA adapter");
    }
    const auto & h = model->hparams;
    if (model->arch != LLM_ARCH_QWEN3NEXT || h.n_layer() == 0 || h.n_layer() > 128 || h.n_expert_groups > 1) {
        throw std::runtime_error("CPU adapter training supports Qwen3Next trunk layers without expert groups");
    }
    if (params.n_tokens == 0 || params.n_tokens > h.n_ctx_train || params.n_tokens > 16384 || params.n_threads <= 0 || params.n_threads > 1024) {
        throw std::runtime_error("invalid complete sequence length or CPU thread count");
    }
    if ((h.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_NONE && h.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_LINEAR) ||
            h.yarn_ext_factor > 0 || h.yarn_attn_factor != 1.0f || h.rope_attn_factor != 1.0f || h.rope_type != LLAMA_ROPE_TYPE_NEOX) {
        throw std::runtime_error("training reference supports unscaled or linearly scaled NeoX RoPE only");
    }
    if (h.ssm_d_conv < 2 || h.ssm_d_conv > 64 || h.ssm_d_state == 0 || h.ssm_d_state > 512 || h.ssm_dt_rank == 0 || h.ssm_dt_rank > 256 ||
            h.ssm_d_inner != h.ssm_d_state*h.ssm_dt_rank || h.ssm_n_group == 0 || h.ssm_dt_rank % h.ssm_n_group != 0) {
        throw std::runtime_error("unsupported gated DeltaNet dimensions");
    }
    for (const auto & entry : model->tensors_by_name) {
        const auto * tensor = entry.second;
        if (!tensor->data || (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) || (tensor->flags & GGML_TENSOR_FLAG_PARAM)) {
            throw std::runtime_error("base weights must be frozen and directly CPU-accessible");
        }
        switch (tensor->type) {
            case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:
                break;
            default: throw std::runtime_error("unsupported base weight format in CPU training reference");
        }
    }
    for (const auto & entry : adapter->ab_map) {
        const auto * weight = model->get_tensor(entry.first.c_str());
        bool supported = false;
        for (uint32_t il = 0; il < h.n_layer(); ++il) {
            const auto & layer = model->layers[il];
            supported |= weight == layer.wo || weight == layer.ssm_out || weight == layer.ffn_gate_shexp ||
                weight == layer.ffn_up_shexp || weight == layer.ffn_down_shexp;
        }
        if (!weight || !supported) throw std::runtime_error("training target must be a trunk attention output or shared-expert projection");
        for (auto * parameter : {entry.second.a, entry.second.b}) {
            if (parameter->type != GGML_TYPE_F32 || !ggml_is_contiguous(parameter) || !parameter->data ||
                    !(parameter->flags & GGML_TENSOR_FLAG_PARAM) || (parameter->buffer && !ggml_backend_buffer_is_host(parameter->buffer))) {
                throw std::runtime_error("adapter training requires initialized CPU F32 parameters");
            }
        }
    }
}

static llama_adapter_training * training_init_graph(llama_model * model, llama_adapter_lora * adapter,
        llama_adapter_training_params params, int layer, char * error, size_t error_capacity, bool allocate = true) {
    if (error && error_capacity) error[0] = '\0';
    try {
        validate_training_model(model, adapter, params);
        const size_t max_nodes = 2048 + 4096*(layer < 0 ? model->hparams.n_layer() : 1) + 8*adapter->ab_map.size();
        const size_t metadata = max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, true) +
            (layer >= 0 ? ggml_graph_overhead_custom(max_nodes, false) : 0);
        if (metadata > params.workspace_budget_bytes) throw std::runtime_error("training graph metadata exceeds workspace budget");
        std::unique_ptr<llama_adapter_training> t(new llama_adapter_training());
        t->model = model;
        t->adapter = adapter;
        t->config = params;
        t->cpu.reset(ggml_backend_cpu_init());
        if (!t->cpu) throw std::runtime_error("cannot initialize CPU training backend");
        t->loras.emplace(adapter, 1.0f);
        t->tokens.resize(params.n_tokens);
        t->positions.resize(params.n_tokens);
        for (uint32_t i = 0; i < params.n_tokens; ++i) t->positions[i] = i;
        t->result.reset(new llm_graph_result(max_nodes, true));
        auto & p = t->graph_params;
        p.arch = model->arch;
        p.hparams = model->hparams;
        p.gtype = LLM_GRAPH_TYPE_ADAPTER_TRAINING;
        p.training_layer = layer;
        p.backend_cpu = t->cpu.get();
        p.cvec = &t->cvec;
        p.loras = &t->loras;
        p.res = t->result.get();
        p.n_outputs = params.n_tokens;
        p.ubatch.b_equal_seqs = true;
        p.ubatch.n_tokens = p.ubatch.n_seq_tokens = params.n_tokens;
        p.ubatch.n_seqs = p.ubatch.n_seqs_unq = p.ubatch.n_pos = 1;
        p.ubatch.token = t->tokens.data();
        p.ubatch.pos = t->positions.data();
        auto & c = p.cparams;
        c.n_ctx = c.n_ctx_seq = c.n_batch = c.n_ubatch = params.n_tokens;
        c.n_seq_max = 1;
        c.causal_attn = true;
        c.pooling_type = LLAMA_POOLING_TYPE_NONE;
        c.rope_freq_base = p.hparams.rope_freq_base_train;
        c.rope_freq_scale = p.hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_NONE ? 1.0f : p.hparams.rope_freq_scale_train;
        c.n_ctx_orig_yarn = p.hparams.n_ctx_orig_yarn ? p.hparams.n_ctx_orig_yarn : p.hparams.n_ctx_train;
        c.yarn_attn_factor = 1.0f;
        c.yarn_beta_fast = p.hparams.yarn_beta_fast;
        c.yarn_beta_slow = p.hparams.yarn_beta_slow;
        auto * graph = model->build_graph(p);
        auto * ctx = t->result->get_ctx();
        if (layer >= 0 && layer < (int) model->hparams.n_layer()) {
            auto * output = t->result->t_training_output;
            t->upstream_tensor = ggml_dup_tensor(ctx, output);
            t->loss = ggml_sum(ctx, ggml_mul(ctx, output, t->upstream_tensor));
            t->forward_context.reset(ggml_init({ggml_graph_overhead_custom(max_nodes, false), nullptr, true}));
            if (!t->forward_context) throw std::runtime_error("cannot allocate forward graph metadata");
            t->forward = ggml_new_graph_custom(t->forward_context.get(), max_nodes, false);
            ggml_build_forward_expand(t->forward, output);
        } else {
            t->labels = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, params.n_tokens);
            t->loss = ggml_cross_entropy_loss_sparse(ctx, t->result->get_logits(), t->labels);
        }
        ggml_set_loss(t->loss);
        ggml_build_forward_expand(graph, t->loss);
        bool has_parameters = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) has_parameters |= (ggml_graph_node(graph, i)->flags & GGML_TENSOR_FLAG_PARAM) != 0;
        // The first layer can be entirely frozen. No derivative before its embedding lookup is required.
        if (has_parameters) ggml_build_backward_expand(ctx, graph, nullptr);
        for (const auto & entry : adapter->ab_map) for (auto * parameter : {entry.second.a, entry.second.b}) {
            auto * gradient = ggml_graph_get_grad(graph, parameter);
            if (!gradient && layer >= 0) continue;
            if (!gradient || gradient->type != GGML_TYPE_F32 || !ggml_is_contiguous(gradient)) throw std::runtime_error("adapter target requires a connected contiguous F32 gradient");
            t->gradients.emplace(parameter->name, gradient);
        }
        if (t->result->t_training_input) {
            t->input_gradient = ggml_graph_get_grad(graph, t->result->t_training_input);
            if (!t->input_gradient || !ggml_is_contiguous(t->input_gradient)) throw std::runtime_error("missing contiguous layer input derivative");
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(t->cpu.get(), ggml_graph_node(graph, i))) {
                throw std::runtime_error(std::string("CPU training operation unsupported: ") + ggml_op_name(ggml_graph_node(graph, i)->op));
            }
        }
        t->plan = ggml_graph_plan(graph, params.n_threads, nullptr);
        t->workspace = checked_add(metadata, ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, ggml_backend_cpu_buffer_type()));
        t->workspace = checked_add(t->workspace, t->plan.work_size);
        t->workspace = checked_add(t->workspace, params.n_tokens*(sizeof(llama_token) + sizeof(llama_pos)));
        const size_t block = (size_t) ceil(sqrt((double) params.n_tokens));
        const size_t slots = (params.n_tokens + block - 1)/block + block + 3;
        t->workspace = checked_add(t->workspace, slots*model->hparams.ssm_d_state*model->hparams.ssm_d_state*sizeof(float));
        if (t->workspace > params.workspace_budget_bytes) throw std::runtime_error("training graph workspace exceeds budget");
        if (!allocate) return t.release();
        t->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()));
        if (!t->buffer) throw std::runtime_error("cannot allocate CPU training workspace");
        ggml_backend_buffer_clear(t->buffer.get(), 0);
        t->scratch.resize(t->plan.work_size);
        t->plan.work_data = t->scratch.data();
        return t.release();
    } catch (const std::exception & e) {
        if (error && error_capacity) snprintf(error, error_capacity, "%s", e.what());
        return nullptr;
    }
}

llama_adapter_training * llama_adapter_training_init(llama_model * model, llama_adapter_lora * adapter,
        llama_adapter_training_params params, char * error, size_t error_capacity) {
    return training_init_graph(model, adapter, params, -1, error, error_capacity);
}

llama_adapter_training * llama_adapter_training_init_checkpointed(llama_model * model, llama_adapter_lora * adapter,
        llama_adapter_training_params params, const llama_adapter_checkpoint_store * store, char * error, size_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    try {
        validate_training_model(model, adapter, params);
        if (store && (!store->write || !store->read || !store->clear)) throw std::runtime_error("checkpoint store requires write, read and clear callbacks");
        const size_t elements = (size_t) model->tok_embd->ne[0]*params.n_tokens;
        if (elements > SIZE_MAX/sizeof(float)/(model->hparams.n_layer()+2)) throw std::runtime_error("layer boundary size overflow");
        const size_t boundary_bytes = elements*sizeof(float);
        const size_t gradient_metadata = 2*adapter->ab_map.size()*ggml_tensor_overhead();
        size_t retained = checked_add(gradient_metadata, boundary_bytes*2);
        if (!store) retained = checked_add(retained, boundary_bytes*(model->hparams.n_layer()-1));
        for (const auto & entry : adapter->ab_map) for (auto * p : {entry.second.a, entry.second.b}) retained = checked_add(retained, ggml_nbytes(p));
        if (retained >= params.workspace_budget_bytes) throw std::runtime_error("layer boundaries and adapter gradients exceed workspace budget");
        auto stage_params = params;
        stage_params.workspace_budget_bytes -= retained;
        size_t peak = 0;
        std::set<std::string> connected;
        for (uint32_t layer = 0; layer <= model->hparams.n_layer(); ++layer) {
            char reason[256];
            std::unique_ptr<llama_adapter_training> stage(training_init_graph(model, adapter, stage_params, layer, reason, sizeof(reason), false));
            if (!stage) throw std::runtime_error(std::string("layer preflight: ")+reason);
            peak = std::max(peak, stage->workspace);
            for (const auto & entry : stage->gradients) {
                if (!connected.insert(entry.first).second) throw std::runtime_error("adapter parameter is shared across training stages");
            }
        }
        if (connected.size() != adapter->ab_map.size()*2) throw std::runtime_error("adapter target is not connected to checkpointed training loss");
        std::unique_ptr<llama_adapter_training> t(new llama_adapter_training());
        t->model = model; t->adapter = adapter; t->config = params; t->checkpointed = true;
        t->retained_workspace = retained; t->workspace = checked_add(retained, peak);
        if (store) t->store = *store;
        t->boundary.resize(elements); t->upstream.resize(elements);
        if (!store) {
            t->boundaries.resize(model->hparams.n_layer()-1);
            for (auto & boundary : t->boundaries) boundary.resize(elements);
        }
        t->gradient_context.reset(ggml_init({gradient_metadata, nullptr, true}));
        if (!t->gradient_context) throw std::runtime_error("cannot allocate adapter gradient metadata");
        for (const auto & entry : adapter->ab_map) for (auto * p : {entry.second.a, entry.second.b}) {
            auto & values = t->gradient_values[p->name];
            values.resize(ggml_nelements(p));
            auto * gradient = ggml_dup_tensor(t->gradient_context.get(), p);
            gradient->data = values.data();
            t->gradients.emplace(p->name, gradient);
        }
        return t.release();
    } catch (const std::exception & e) {
        if (error && error_capacity) snprintf(error, error_capacity, "%s", e.what());
        return nullptr;
    }
}

static float compute_checkpointed(llama_adapter_training * t, const llama_token * tokens, const llama_token * labels,
        ggml_abort_callback abort_callback, void * abort_data) {
    const size_t bytes = t->boundary.size()*sizeof(float);
    const uint32_t layers = t->model->hparams.n_layer();
    auto stage_params = t->config;
    stage_params.workspace_budget_bytes -= t->retained_workspace;
    auto cancelled = [&]() { if (abort_callback && abort_callback(abort_data)) throw std::runtime_error("training compute cancelled"); };
    auto read = [&](uint32_t boundary) {
        if (t->store.read) {
            if (!t->store.read(t->store.user_data, boundary, t->boundary.data(), bytes)) throw std::runtime_error("activation checkpoint read failed");
        } else {
            memcpy(t->boundary.data(), t->boundaries.at(boundary-1).data(), bytes);
        }
        for (float value : t->boundary) if (!std::isfinite(value)) throw std::runtime_error("nonfinite activation checkpoint");
    };
    auto build = [&](uint32_t layer) {
        cancelled();
        char reason[256];
        std::unique_ptr<llama_adapter_training> stage(training_init_graph(t->model, t->adapter, stage_params, layer, reason, sizeof(reason)));
        if (!stage) throw std::runtime_error(std::string("layer execution: ")+reason);
        std::copy(tokens, tokens+t->config.n_tokens, stage->tokens.begin());
        stage->result->set_inputs(&stage->graph_params.ubatch);
        if (layer > 0) ggml_backend_tensor_set(stage->result->t_training_input, t->boundary.data(), 0, bytes);
        stage->plan.abort_callback = abort_callback;
        stage->plan.abort_callback_data = abort_data;
        return stage;
    };
    auto execute = [&](llama_adapter_training * stage, bool backward) {
        cancelled();
        ggml_graph_reset(stage->result->get_gf());
        const auto status = ggml_graph_compute(backward ? stage->result->get_gf() : stage->forward, &stage->plan);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error(status == GGML_STATUS_ABORTED ? "training compute cancelled" : "training layer compute failed");
    };
    auto copy_gradients = [&](llama_adapter_training * stage) {
        if (stage->input_gradient) ggml_backend_tensor_get(stage->input_gradient, t->upstream.data(), 0, bytes);
        for (const auto & entry : stage->gradients) {
            auto * destination = t->gradients.at(entry.first);
            ggml_backend_tensor_get(entry.second, destination->data, 0, ggml_nbytes(destination));
        }
    };
    for (uint32_t layer = 0; layer < layers; ++layer) {
        auto stage = build(layer);
        execute(stage.get(), false);
        ggml_backend_tensor_get(stage->result->t_training_output, t->boundary.data(), 0, bytes);
        for (float value : t->boundary) if (!std::isfinite(value)) throw std::runtime_error("nonfinite layer boundary");
        if (layer+1 < layers) {
            if (t->store.write) {
                if (!t->store.write(t->store.user_data, layer+1, t->boundary.data(), bytes)) throw std::runtime_error("activation checkpoint write failed");
            } else {
                memcpy(t->boundaries[layer].data(), t->boundary.data(), bytes);
            }
        }
    }
    float loss;
    {
        auto head = build(layers);
        ggml_backend_tensor_set(head->labels, labels, 0, t->config.n_tokens*sizeof(llama_token));
        execute(head.get(), true);
        loss = *(float *) head->loss->data;
        copy_gradients(head.get());
    }
    for (int layer = (int) layers-1; layer >= 0; --layer) {
        cancelled();
        if (layer > 0) read(layer);
        auto stage = build(layer);
        ggml_backend_tensor_set(stage->upstream_tensor, t->upstream.data(), 0, bytes);
        execute(stage.get(), true);
        copy_gradients(stage.get());
    }
    return loss;
}

void llama_adapter_training_free(llama_adapter_training * training) { delete training; }

size_t llama_adapter_training_workspace_size(const llama_adapter_training * training) { return training ? training->workspace : 0; }

size_t llama_adapter_training_checkpoint_size(const llama_adapter_training * training) {
    return training && training->checkpointed ? training->boundary.size()*sizeof(float)*(training->model->hparams.n_layer()-1) : 0;
}

bool llama_adapter_training_compute(llama_adapter_training * t, const llama_token * tokens, const llama_token * labels,
        float * loss, ggml_abort_callback abort_callback, void * abort_data) {
    if (!t) return false;
    t->valid = false;
    t->consumed = false;
    t->error.clear();
    try {
        if (t->store.clear && !t->store.clear(t->store.user_data)) throw std::runtime_error("activation checkpoint cleanup failed");
        if (!tokens || !labels || !loss) throw std::runtime_error("missing training input or loss destination");
        const int64_t vocab = t->model->tok_embd->ne[1];
        size_t supervised = 0;
        for (uint32_t i = 0; i < t->config.n_tokens; ++i) {
            if (tokens[i] < 0 || tokens[i] >= vocab || labels[i] < -1 || labels[i] >= vocab) throw std::runtime_error("invalid training token or label");
            supervised += labels[i] != -1;
        }
        if (!supervised) throw std::runtime_error("training sequence has no supervised tokens");
        if (abort_callback && abort_callback(abort_data)) throw std::runtime_error("training compute cancelled");
        float value;
        if (t->checkpointed) {
            value = compute_checkpointed(t, tokens, labels, abort_callback, abort_data);
        } else {
            std::copy(tokens, tokens + t->config.n_tokens, t->tokens.begin());
            t->result->set_inputs(&t->graph_params.ubatch);
            ggml_backend_tensor_set(t->labels, labels, 0, t->config.n_tokens*sizeof(llama_token));
            ggml_graph_reset(t->result->get_gf());
            t->plan.abort_callback = abort_callback;
            t->plan.abort_callback_data = abort_data;
            const auto status = ggml_graph_compute(t->result->get_gf(), &t->plan);
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error(status == GGML_STATUS_ABORTED ? "training compute cancelled" : "training graph compute failed");
            value = *(const float *) t->loss->data;
        }
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite training loss");
        for (const auto & entry : t->gradients) {
            const float * values = (const float *) entry.second->data;
            for (int64_t i = 0; i < ggml_nelements(entry.second); ++i) if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite adapter gradient");
        }
        if (t->store.clear && !t->store.clear(t->store.user_data)) throw std::runtime_error("activation checkpoint cleanup failed");
        *loss = value;
        t->supervised = supervised;
        t->revision = t->adapter->training_revision;
        t->valid = true;
        return true;
    } catch (const std::exception & e) {
        t->error = e.what();
        if (t->store.clear && !t->store.clear(t->store.user_data)) t->error += "; activation checkpoint cleanup failed";
        return false;
    }
}

size_t llama_adapter_training_gradient(const llama_adapter_training * t, const char * name, float * data, size_t capacity) {
    if (!t || !t->valid || t->revision != t->adapter->training_revision || !name) return 0;
    const auto found = t->gradients.find(name);
    if (found == t->gradients.end()) return 0;
    const size_t count = ggml_nelements(found->second);
    if (data && capacity >= count) memcpy(data, found->second->data, count*sizeof(float));
    return count;
}

const char * llama_adapter_training_error(const llama_adapter_training * training) { return training ? training->error.c_str() : "null training context"; }

struct llama_adapter_optimizer {
    struct parameter {
        ggml_tensor * weight;
        std::vector<float> m, v, sum;
    };
    llama_adapter_lora * adapter;
    llama_adapter_optimizer_params config;
    std::map<std::string, parameter> parameters;
    std::string binding, error;
    uint64_t steps = 0, tokens = 0;
    uint64_t revision = 0;
};

llama_adapter_optimizer_params llama_adapter_optimizer_default_params() {
    return {0.9f, 0.999f, 1e-8f, 0.01f, 1.0f, 256*1024*1024};
}

llama_adapter_optimizer * llama_adapter_optimizer_init(llama_adapter_lora * adapter, llama_adapter_optimizer_params params,
        const char * binding, char * error, size_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    try {
        if (!adapter || adapter->ab_map.empty() || !std::isfinite(adapter->alpha) || adapter->alpha <= 0 ||
                !binding || strlen(binding) != 64 || strspn(binding, "0123456789abcdef") != 64) {
            throw std::runtime_error("optimizer requires an initialized adapter and lowercase SHA-256 manifest binding");
        }
        if (!std::isfinite(params.beta1) || params.beta1 < 0 || params.beta1 >= 1 ||
                !std::isfinite(params.beta2) || params.beta2 < 0 || params.beta2 >= 1 ||
                !std::isfinite(params.epsilon) || params.epsilon <= 0 ||
                !std::isfinite(params.weight_decay) || params.weight_decay < 0 ||
                !std::isfinite(params.gradient_clip) || params.gradient_clip <= 0) throw std::runtime_error("invalid AdamW configuration");
        size_t required = 0;
        for (const auto & entry : adapter->ab_map) for (auto * tensor : {entry.second.a, entry.second.b}) {
            if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !tensor->data ||
                    !(tensor->flags & GGML_TENSOR_FLAG_PARAM) || (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer))) {
                throw std::runtime_error("optimizer requires contiguous trainable CPU F32 parameters");
            }
            required = checked_add(required, checked_add(ggml_nbytes(tensor), checked_add(ggml_nbytes(tensor), ggml_nbytes(tensor))));
        }
        if (required > params.state_budget_bytes) throw std::runtime_error("adapter optimizer state exceeds budget");
        std::unique_ptr<llama_adapter_optimizer> result(new llama_adapter_optimizer());
        result->adapter = adapter; result->config = params; result->binding = binding;
        result->revision = adapter->training_revision;
        for (const auto & entry : adapter->ab_map) for (auto * tensor : {entry.second.a, entry.second.b}) {
            const size_t count = ggml_nelements(tensor);
            result->parameters.emplace(tensor->name, llama_adapter_optimizer::parameter{tensor,
                std::vector<float>(count), std::vector<float>(count), std::vector<float>(count)});
        }
        return result.release();
    } catch (const std::exception & e) {
        if (error && error_capacity) snprintf(error, error_capacity, "%s", e.what());
        return nullptr;
    }
}

void llama_adapter_optimizer_free(llama_adapter_optimizer * optimizer) { delete optimizer; }
uint64_t llama_adapter_optimizer_step_count(const llama_adapter_optimizer * optimizer) { return optimizer ? optimizer->steps : 0; }
uint64_t llama_adapter_optimizer_pending_tokens(const llama_adapter_optimizer * optimizer) { return optimizer ? optimizer->tokens : 0; }
const char * llama_adapter_optimizer_error(const llama_adapter_optimizer * optimizer) { return optimizer ? optimizer->error.c_str() : "null optimizer"; }

bool llama_adapter_optimizer_accumulate(llama_adapter_optimizer * optimizer, llama_adapter_training * training) {
    if (!optimizer) return false;
    optimizer->error.clear();
    try {
        if (!training || training->adapter != optimizer->adapter || !training->valid || training->consumed ||
                optimizer->revision != optimizer->adapter->training_revision || training->revision != optimizer->revision ||
                !training->supervised || training->supervised > UINT64_MAX - optimizer->tokens) {
            throw std::runtime_error("optimizer requires fresh unconsumed gradients for its adapter");
        }
        for (const auto & entry : optimizer->parameters) {
            const auto * grad = training->gradients.at(entry.first);
            if (!ggml_is_contiguous(grad) || ggml_nelements(grad) != (int64_t) entry.second.sum.size()) throw std::runtime_error("incompatible gradient layout");
            const float * values = (const float *) grad->data;
            for (size_t i = 0; i < entry.second.sum.size(); ++i) {
                const float candidate = (float) (entry.second.sum[i] + (double) values[i]*training->supervised);
                if (!std::isfinite(candidate)) throw std::runtime_error("nonfinite accumulated gradient");
            }
        }
        for (auto & entry : optimizer->parameters) {
            const float * values = (const float *) training->gradients.at(entry.first)->data;
            for (size_t i = 0; i < entry.second.sum.size(); ++i) entry.second.sum[i] = (float) (entry.second.sum[i] + (double) values[i]*training->supervised);
        }
        optimizer->tokens += training->supervised;
        training->consumed = true;
        return true;
    } catch (const std::exception & e) { optimizer->error = e.what(); return false; }
}

bool llama_adapter_optimizer_step(llama_adapter_optimizer * optimizer, float learning_rate) {
    if (!optimizer) return false;
    optimizer->error.clear();
    try {
        if (!optimizer->tokens || optimizer->steps == UINT64_MAX || optimizer->adapter->training_revision == UINT64_MAX ||
                optimizer->revision != optimizer->adapter->training_revision || !std::isfinite(learning_rate) || learning_rate <= 0) {
            throw std::runtime_error("AdamW step requires accumulated tokens and a positive finite learning rate");
        }
        const auto & c = optimizer->config;
        double squared_norm = 0;
        for (const auto & entry : optimizer->parameters) for (float sum : entry.second.sum) {
            const double g = sum/(double) optimizer->tokens;
            squared_norm += g*g;
        }
        const double norm = sqrt(squared_norm);
        if (!std::isfinite(norm)) throw std::runtime_error("nonfinite global gradient norm");
        const double scale = (norm > c.gradient_clip ? c.gradient_clip/norm : 1.0)/(double) optimizer->tokens;
        const double bias1 = 1.0 - pow((double) c.beta1, (double) (optimizer->steps+1));
        const double bias2 = 1.0 - pow((double) c.beta2, (double) (optimizer->steps+1));
        // The first pass validates every candidate. The second is a non-cancellable commit.
        for (int commit = 0; commit < 2; ++commit) for (auto & entry : optimizer->parameters) {
            auto & p = entry.second;
            auto * values = (float *) p.weight->data;
            for (size_t i = 0; i < p.sum.size(); ++i) {
                const double g = p.sum[i]*scale;
                const float m = (float) (c.beta1*p.m[i] + (1.0-c.beta1)*g);
                const float v = (float) (c.beta2*p.v[i] + (1.0-c.beta2)*g*g);
                const float value = (float) (values[i]*(1.0-learning_rate*c.weight_decay) - learning_rate*(m/bias1)/(sqrt(v/bias2)+c.epsilon));
                if (!commit && (!std::isfinite(m) || !std::isfinite(v) || !std::isfinite(value))) throw std::runtime_error("nonfinite AdamW candidate; step not committed");
                if (commit) { values[i] = value; p.m[i] = m; p.v[i] = v; p.sum[i] = 0; }
            }
        }
        optimizer->tokens = 0;
        ++optimizer->steps;
        optimizer->revision = ++optimizer->adapter->training_revision;
        return true;
    } catch (const std::exception & e) { optimizer->error = e.what(); return false; }
}

size_t llama_adapter_optimizer_save(llama_adapter_optimizer * optimizer, void * data, size_t capacity) {
    if (!optimizer) return 0;
    optimizer->error.clear();
    try {
        if (optimizer->revision != optimizer->adapter->training_revision) throw std::runtime_error("optimizer state is stale after another parameter commit");
        gguf_context_ptr meta(gguf_init_empty());
        ggml_context_ptr ctx(ggml_init({4*optimizer->parameters.size()*ggml_tensor_overhead(), nullptr, true}));
        if (!meta || !ctx) throw std::runtime_error("cannot allocate optimizer checkpoint metadata");
        gguf_set_val_str(meta.get(), "general.type", "adapter_optimizer");
        gguf_set_val_u32(meta.get(), "optimizer.version", 1);
        gguf_set_val_str(meta.get(), "general.architecture", llm_arch_name(optimizer->adapter->model->arch));
        gguf_set_val_str(meta.get(), "optimizer.binding.sha256", optimizer->binding.c_str());
        gguf_set_val_f32(meta.get(), "adapter.lora.alpha", optimizer->adapter->alpha);
        gguf_set_val_f32(meta.get(), "optimizer.beta1", optimizer->config.beta1);
        gguf_set_val_f32(meta.get(), "optimizer.beta2", optimizer->config.beta2);
        gguf_set_val_f32(meta.get(), "optimizer.epsilon", optimizer->config.epsilon);
        gguf_set_val_f32(meta.get(), "optimizer.weight_decay", optimizer->config.weight_decay);
        gguf_set_val_f32(meta.get(), "optimizer.gradient_clip", optimizer->config.gradient_clip);
        gguf_set_val_u64(meta.get(), "optimizer.steps", optimizer->steps);
        gguf_set_val_u64(meta.get(), "optimizer.pending_tokens", optimizer->tokens);
        size_t index = 0;
        for (const auto & entry : optimizer->parameters) {
            const auto & p = entry.second;
            const std::string prefix = "parameter." + std::to_string(index++);
            gguf_set_val_str(meta.get(), (prefix+".name").c_str(), entry.first.c_str());
            const void * sources[] = {p.weight->data, p.m.data(), p.v.data(), p.sum.data()};
            int source = 0;
            for (const char * suffix : {"value", "m", "v", "sum"}) {
                auto * tensor = ggml_dup_tensor(ctx.get(), p.weight);
                ggml_set_name(tensor, (prefix+"."+suffix).c_str());
                tensor->data = const_cast<void *>(sources[source++]);
                for (size_t i = 0; i < p.sum.size(); ++i) if (!std::isfinite(((const float *) tensor->data)[i])) throw std::runtime_error("nonfinite checkpoint payload");
                gguf_add_tensor(meta.get(), tensor);
            }
        }
        const size_t header = gguf_get_meta_size(meta.get());
        size_t size = header;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) size = std::max(size, checked_add(header, checked_add(gguf_get_tensor_offset(meta.get(), i), gguf_get_tensor_size(meta.get(), i))));
        if (!data || capacity < size) return size;
        memset(data, 0, size);
        gguf_get_meta_data(meta.get(), data);
        for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
            const size_t offset = header + gguf_get_tensor_offset(meta.get(), gguf_find_tensor(meta.get(), tensor->name));
            memcpy((char *) data + offset, tensor->data, ggml_nbytes(tensor));
        }
        return size;
    } catch (const std::exception & e) { optimizer->error = e.what(); return 0; }
}

bool llama_adapter_optimizer_load(llama_adapter_optimizer * optimizer, const void * data, size_t size) {
    if (!optimizer) return false;
    optimizer->error.clear();
    try {
        if (!data || !size || optimizer->adapter->training_revision == UINT64_MAX) throw std::runtime_error("empty optimizer checkpoint or exhausted parameter revision");
        ggml_context * raw = nullptr;
        gguf_context_ptr meta(gguf_init_from_buffer(data, size, {true, &raw}));
        ggml_context_ptr ctx(raw);
        if (!meta || !ctx) throw std::runtime_error("invalid optimizer checkpoint GGUF");
        auto key = [&](const char * name, gguf_type type) {
            const int64_t id = gguf_find_key(meta.get(), name);
            if (id < 0 || gguf_get_kv_type(meta.get(), id) != type) throw std::runtime_error(std::string("invalid checkpoint metadata: ")+name);
            return id;
        };
        auto string_equal = [&](const char * name, const char * expected) { return strcmp(gguf_get_val_str(meta.get(), key(name, GGUF_TYPE_STRING)), expected) == 0; };
        auto float_equal = [&](const char * name, float expected) { return gguf_get_val_f32(meta.get(), key(name, GGUF_TYPE_FLOAT32)) == expected; };
        if (!string_equal("general.type", "adapter_optimizer") || gguf_get_val_u32(meta.get(), key("optimizer.version", GGUF_TYPE_UINT32)) != 1 ||
                !string_equal("general.architecture", llm_arch_name(optimizer->adapter->model->arch)) ||
                !string_equal("optimizer.binding.sha256", optimizer->binding.c_str()) || !float_equal("adapter.lora.alpha", optimizer->adapter->alpha) ||
                !float_equal("optimizer.beta1", optimizer->config.beta1) || !float_equal("optimizer.beta2", optimizer->config.beta2) ||
                !float_equal("optimizer.epsilon", optimizer->config.epsilon) || !float_equal("optimizer.weight_decay", optimizer->config.weight_decay) ||
                !float_equal("optimizer.gradient_clip", optimizer->config.gradient_clip) || gguf_get_n_tensors(meta.get()) != (int64_t) optimizer->parameters.size()*4) {
            throw std::runtime_error("checkpoint binding, configuration or parameter count mismatch");
        }
        const uint64_t steps = gguf_get_val_u64(meta.get(), key("optimizer.steps", GGUF_TYPE_UINT64));
        const uint64_t tokens = gguf_get_val_u64(meta.get(), key("optimizer.pending_tokens", GGUF_TYPE_UINT64));
        struct copy { void * target; const char * source; size_t bytes; };
        std::vector<copy> copies;
        size_t index = 0;
        for (auto & entry : optimizer->parameters) {
            auto & p = entry.second;
            const std::string prefix = "parameter." + std::to_string(index++);
            if (!string_equal((prefix+".name").c_str(), entry.first.c_str())) throw std::runtime_error("checkpoint target names differ");
            void * destinations[] = {p.weight->data, p.m.data(), p.v.data(), p.sum.data()};
            int destination = 0;
            for (const char * suffix : {"value", "m", "v", "sum"}) {
                const std::string name = prefix+"."+suffix;
                auto * tensor = ggml_get_tensor(ctx.get(), name.c_str());
                const int64_t id = gguf_find_tensor(meta.get(), name.c_str());
                if (!tensor || id < 0 || tensor->type != GGML_TYPE_F32 || !ggml_are_same_shape(tensor, p.weight)) throw std::runtime_error("checkpoint parameter shape/type mismatch");
                const size_t offset = checked_add(gguf_get_data_offset(meta.get()), gguf_get_tensor_offset(meta.get(), id));
                if (offset > size || ggml_nbytes(tensor) > size-offset) throw std::runtime_error("truncated checkpoint tensor");
                const char * source = (const char *) data+offset;
                for (size_t i = 0; i < p.sum.size(); ++i) {
                    float value;
                    memcpy(&value, source+i*sizeof(float), sizeof(float));
                    if (!std::isfinite(value) || (destination == 2 && value < 0) || (destination == 3 && tokens == 0 && value != 0)) throw std::runtime_error("invalid checkpoint optimizer payload");
                }
                copies.push_back({destinations[destination++], source, ggml_nbytes(tensor)});
            }
        }
        for (const auto & copy : copies) memcpy(copy.target, copy.source, copy.bytes);
        optimizer->steps = steps; optimizer->tokens = tokens;
        optimizer->revision = ++optimizer->adapter->training_revision;
        return true;
    } catch (const std::exception & e) { optimizer->error = e.what(); return false; }
}
