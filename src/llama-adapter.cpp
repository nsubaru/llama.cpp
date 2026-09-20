#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <map>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer());
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer(); il++) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer(); il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static void llama_adapter_lora_init_impl(llama_model & model, FILE * file, llama_adapter_lora & adapter,
        const void * data = nullptr, size_t data_size = 0) {
    ggml_context * ctx_init = nullptr;
    gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { data
        ? gguf_init_from_buffer(data, data_size, meta_gguf_params)
        : gguf_init_from_file_ptr(file, meta_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter from file");
    }

    ggml_context_ptr ctx { ctx_init };

    // must come after gguf_init_from_file_ptr, the llama_file constructor moves the file position
    std::unique_ptr<llama_file> gguf_file;
    if (file) {
        gguf_file = std::make_unique<llama_file>(file);
        data_size = gguf_file->size();
    }

    // check metadata
    {
        const gguf_context * gguf_ctx = ctx_gguf.get();

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const std::string value = gguf_kv_to_str(gguf_ctx, i);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            const size_t MAX_VALUE_LEN = 40;
            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            if (id >= 0 && gguf_get_kv_type(gguf_ctx, id) != GGUF_TYPE_STRING) {
                throw std::runtime_error("invalid metadata type: " + key);
            }
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            if (id >= 0 && gguf_get_kv_type(gguf_ctx, id) != GGUF_TYPE_FLOAT32) {
                throw std::runtime_error("invalid metadata type: " + key);
            }
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));
        if (!std::isfinite(adapter.alpha) || adapter.alpha < 0) {
            throw std::runtime_error("invalid LoRA alpha");
        }

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(ctx_gguf.get(), kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
            const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf.get());
    if (n_tensors == 0) {
        throw std::runtime_error("empty LoRA adapter");
    }

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            if (data) {
                throw std::runtime_error("unsupported LoRA normalization tensor: " + name);
            }
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }
        if (data && ((w.a->type != GGML_TYPE_F32 && w.a->type != GGML_TYPE_F16) ||
                     (w.b->type != GGML_TYPE_F32 && w.b->type != GGML_TYPE_F16))) {
            throw std::runtime_error("adapter tensors must use F32 or F16: " + name);
        }

        // device buft and device ctx
        const auto * model_tensor = model.get_tensor(name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        if (!dev_ctx) {
            throw std::runtime_error("failed to allocate adapter metadata");
        }
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[0] != w.b->ne[0] || w.a->ne[2] != 1 || w.b->ne[2] != 1 ||
                w.a->ne[3] != 1 || w.b->ne[3] != 1) {
                throw std::runtime_error("invalid embedding adapter rank or dimensions");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
            if (w.a->ne[2] != model_tensor->ne[2] || w.b->ne[2] != model_tensor->ne[2] ||
                w.a->ne[3] != model_tensor->ne[3] || w.b->ne[3] != model_tensor->ne[3]) {
                throw std::runtime_error("LoRA expert dimensions do not match the base tensor: " + name);
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            const size_t header = gguf_get_data_offset(ctx_gguf.get());
            const size_t relative = gguf_get_tensor_offset(ctx_gguf.get(), gguf_find_tensor(ctx_gguf.get(), orig->name));
            if (header > data_size || relative > data_size - header) {
                throw std::runtime_error("adapter tensor offset exceeds buffer");
            }
            const size_t offs = header + relative;
            const size_t size = ggml_nbytes(orig);
            if (offs > data_size || size > data_size - offs) {
                throw std::runtime_error(format("LoRA tensor '%s' data is not within the file bounds, file is corrupted or incomplete", orig->name));
            }
            const void * payload;
            if (data) {
                payload = static_cast<const uint8_t *>(data) + offs;
                if (!ggml_validate_row_data(orig->type, payload, size)) {
                    throw std::runtime_error("invalid adapter tensor values: " + std::string(orig->name));
                }
            } else {
                read_buf.resize(size);
                gguf_file->seek(offs, SEEK_SET);
                gguf_file->read_raw(read_buf.data(), size);
                payload = read_buf.data();
            }
            ggml_backend_tensor_set(dev, payload, 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    adapter.export_metadata.reset(gguf_init_empty());
    if (!adapter.export_metadata) throw std::runtime_error("cannot retain adapter metadata");
    gguf_set_kv(adapter.export_metadata.get(), ctx_gguf.get());

    // register adapter with model
    model.loras.insert(&adapter);

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    FILE * file = ggml_fopen(path_lora, "rb");
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path_lora, strerror(errno));
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    fclose(file);

    return adapter;
}

llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(llama_model * model, FILE * file) {
    if (!model || !file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }

    llama_adapter_lora * adapter = new llama_adapter_lora(model);

    try {
        llama_adapter_lora_init_impl(*model, file, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

llama_adapter_lora * llama_adapter_lora_init_training(llama_model * model, const char * const * targets,
        size_t n_targets, uint32_t rank, float alpha, uint64_t seed) {
    try {
        if (!model || !targets || !n_targets || !rank || !std::isfinite(alpha) || alpha <= 0) {
            throw std::runtime_error("invalid training adapter configuration");
        }
        std::map<std::string, const ggml_tensor *> selected;
        for (size_t i = 0; i < n_targets; ++i) {
            if (!targets[i] || strlen(targets[i]) + strlen(".lora_a") >= GGML_MAX_NAME) {
                throw std::runtime_error("invalid adapter target name");
            }
            const auto * weight = model->get_tensor(targets[i]);
            if (!weight || !ggml_is_matrix(weight) || weight->ne[1] <= 1 ||
                    rank > std::min(weight->ne[0], weight->ne[1]) || strcmp(targets[i], "token_embd.weight") == 0) {
                throw std::runtime_error("training adapters require supported 2D projection targets");
            }
            if (!selected.emplace(targets[i], weight).second) {
                throw std::runtime_error("duplicate adapter target");
            }
        }
        auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) throw std::runtime_error("no CPU backend for training parameters");
        std::unique_ptr<llama_adapter_lora> adapter(new llama_adapter_lora(model));
        adapter->alpha = alpha;
        adapter->gguf_kv["general.architecture"] = llm_arch_name(model->arch);
        adapter->gguf_kv["general.type"] = "adapter";
        adapter->gguf_kv["adapter.type"] = "lora";
        adapter->gguf_kv["adapter.training.initialization"] = "uniform-splitmix64-v1";
        adapter->gguf_kv["adapter.training.seed"] = std::to_string(seed);
        if (n_targets > SIZE_MAX / (2*ggml_tensor_overhead())) throw std::runtime_error("too many adapter targets");
        ggml_context_ptr ctx(ggml_init({2*n_targets*ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) throw std::runtime_error("cannot allocate adapter metadata");
        for (auto & entry : selected) {
            auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, entry.second->ne[0], rank);
            auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rank, entry.second->ne[1]);
            ggml_set_name(a, (entry.first + ".lora_a").c_str());
            ggml_set_name(b, (entry.first + ".lora_b").c_str());
            ggml_set_param(a);
            ggml_set_param(b);
            adapter->ab_map.emplace(entry.first, llama_adapter_lora_weight(a, b));
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(cpu)));
        if (!buffer) throw std::runtime_error("cannot allocate adapter parameters");
        ggml_backend_buffer_clear(buffer.get(), 0);
        uint64_t state = seed;
        for (const auto & entry : selected) {
            auto * a = adapter->ab_map.at(entry.first).a;
            const double bound = sqrt(3.0 / a->ne[0]);
            auto * values = static_cast<float *>(a->data);
            for (int64_t i = 0; i < ggml_nelements(a); ++i) {
                uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
                value = (value ^ (value >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
                value = (value ^ (value >> 27))*UINT64_C(0x94d049bb133111eb);
                value ^= value >> 31;
                values[i] = static_cast<float>((2.0 * (value >> 11) / 9007199254740992.0 - 1.0)*bound);
            }
        }
        adapter->ctxs.emplace_back(std::move(ctx));
        adapter->bufs.emplace_back(std::move(buffer));
        model->loras.insert(adapter.get());
        return adapter.release();
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return nullptr;
    }
}

size_t llama_adapter_lora_export_buffer(const llama_adapter_lora * adapter, void * data, size_t capacity) {
    try {
        if (!adapter || adapter->ab_map.empty() || !std::isfinite(adapter->alpha) || adapter->alpha < 0) {
            throw std::runtime_error("invalid adapter for export");
        }
        gguf_context_ptr meta(gguf_init_empty());
        ggml_context_ptr ctx(ggml_init({2*adapter->ab_map.size()*ggml_tensor_overhead(), nullptr, true}));
        if (!meta || !ctx) throw std::runtime_error("cannot allocate export metadata");
        if (adapter->export_metadata) gguf_set_kv(meta.get(), adapter->export_metadata.get());
        const std::map<std::string, std::string> metadata(adapter->gguf_kv.begin(), adapter->gguf_kv.end());
        for (const auto & entry : metadata) {
            if (gguf_find_key(meta.get(), entry.first.c_str()) < 0) {
                gguf_set_val_str(meta.get(), entry.first.c_str(), entry.second.c_str());
            }
        }
        gguf_set_val_str(meta.get(), "general.architecture", llm_arch_name(adapter->model->arch));
        gguf_set_val_str(meta.get(), "general.type", "adapter");
        gguf_set_val_str(meta.get(), "adapter.type", "lora");
        gguf_set_val_f32(meta.get(), "adapter.lora.alpha", adapter->alpha);
        if (gguf_find_key(meta.get(), "general.file_type") >= 0) gguf_set_val_u32(meta.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_F16);
        if (!adapter->alora_invocation_tokens.empty()) {
            gguf_set_arr_data(meta.get(), "adapter.alora.invocation_tokens", GGUF_TYPE_UINT32,
                adapter->alora_invocation_tokens.data(), adapter->alora_invocation_tokens.size());
        }
        std::map<std::string, const ggml_tensor *> ordered;
        for (const auto & entry : adapter->ab_map) {
            ordered.emplace(entry.second.a->name, entry.second.a);
            ordered.emplace(entry.second.b->name, entry.second.b);
        }
        for (const auto & entry : ordered) {
            const auto * source = entry.second;
            if ((source->type != GGML_TYPE_F32 && source->type != GGML_TYPE_F16) || !ggml_is_contiguous(source)) {
                throw std::runtime_error("export requires contiguous F32 or F16 adapter tensors");
            }
            auto * target = ggml_new_tensor(ctx.get(), GGML_TYPE_F16, GGML_MAX_DIMS, source->ne);
            ggml_set_name(target, source->name);
            gguf_add_tensor(meta.get(), target);
        }
        const size_t header = gguf_get_meta_size(meta.get());
        size_t required = header;
        for (auto * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
            const size_t offset = gguf_get_tensor_offset(meta.get(), gguf_find_tensor(meta.get(), t->name));
            if (offset > SIZE_MAX-header || ggml_nbytes(t) > SIZE_MAX-header-offset) throw std::runtime_error("adapter export size overflow");
            required = std::max(required, header + offset + ggml_nbytes(t));
        }
        if (!data || capacity < required) return required;
        memset(data, 0, required);
        gguf_get_meta_data(meta.get(), data);
        float values[4096];
        ggml_fp16_t encoded[4096];
        for (const auto & entry : ordered) {
            const auto * source = entry.second;
            auto * output = static_cast<uint8_t *>(data) + header + gguf_get_tensor_offset(meta.get(), gguf_find_tensor(meta.get(), entry.first.c_str()));
            for (int64_t first = 0; first < ggml_nelements(source); first += 4096) {
                const size_t n = std::min<int64_t>(4096, ggml_nelements(source)-first);
                if (source->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(source, values, first*sizeof(float), n*sizeof(float));
                    ggml_fp32_to_fp16_row(values, encoded, n);
                } else {
                    ggml_backend_tensor_get(source, encoded, first*sizeof(ggml_fp16_t), n*sizeof(ggml_fp16_t));
                }
                if (!ggml_validate_row_data(GGML_TYPE_F16, encoded, n*sizeof(ggml_fp16_t))) throw std::runtime_error("adapter values cannot be exported as finite F16");
                memcpy(output + first*sizeof(ggml_fp16_t), encoded, n*sizeof(ggml_fp16_t));
            }
        }
        return required;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, error.what());
        return 0;
    }
}

llama_adapter_lora * llama_adapter_lora_init_from_buffer(llama_model * model, const void * data, size_t size) {
    if (!model || !data || size < 24) {
        return nullptr;
    }
    std::unique_ptr<llama_adapter_lora> adapter(new llama_adapter_lora(model));
    try {
        llama_adapter_lora_init_impl(*model, nullptr, *adapter, data, size);
        return adapter.release();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return nullptr;
    }
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return;
    }

    if (adapter->model != nullptr) {
        adapter->model->loras.erase(adapter);
        adapter->model = nullptr;
    }

    delete adapter;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}
