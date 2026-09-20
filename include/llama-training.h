#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_adapter_training;
    struct llama_adapter_optimizer;

    // Experimental CPU training for one complete causal Qwen3Next sequence.
    // Use the checkpointed initializer for layer recomputation and optional external activation storage.
    struct llama_adapter_training_params {
        uint32_t n_tokens;             // Fixed sequence length; no truncation or padding is performed.
        int32_t n_threads;             // Positive CPU worker count.
        size_t workspace_budget_bytes; // Graph metadata, tensor buffers and compute scratch; excludes borrowed model/adapter and C++ container overhead.
    };

    // Create a full reference forward/backward graph, retaining all intermediate activations.
    // Uses zero initial recurrent state and no inference caches.
    // Supports CPU-accessible frozen weights and an F32 adapter from llama_adapter_lora_init_training.
    // The model and adapter must outlive the returned context. No other context may use or modify the adapter concurrently.
    // Returns NULL on unsupported configuration or insufficient workspace; error is optional and always terminated when capacity > 0.
    LLAMA_API struct llama_adapter_training * llama_adapter_training_init(
            struct llama_model * model,
            struct llama_adapter_lora * adapter,
            struct llama_adapter_training_params params,
            char * error, size_t error_capacity);

    // Caller-owned temporary storage for F32 layer boundaries, numbered 1..n_layer-1.
    // Callbacks run synchronously, must preserve exact bytes, and return false on failure without throwing.
    // clear removes only this context's temporary artifacts; it runs before and after each compute, including failures.
    // Storage capacity, integrity checks, staging memory and I/O cancellation are the caller's responsibility.
    struct llama_adapter_checkpoint_store {
        void * user_data;
        bool (*write)(void * user_data, uint32_t boundary, const void * data, size_t size);
        bool (*read)(void * user_data, uint32_t boundary, void * data, size_t size);
        bool (*clear)(void * user_data);
    };

    // Recompute one complete layer at a time during backward; no sequence or attention context is shortened.
    // A null store retains only layer boundaries in RAM and charges them to the workspace budget.
    // An external store permits disk-backed boundaries; its memory and disk budgets are outside this API's workspace count.
    // Preflight checks every layer/head workspace. Model/adapter/callback lifetimes and exclusive access remain caller-owned.
    LLAMA_API struct llama_adapter_training * llama_adapter_training_init_checkpointed(
            struct llama_model * model, struct llama_adapter_lora * adapter,
            struct llama_adapter_training_params params, const struct llama_adapter_checkpoint_store * store,
            char * error, size_t error_capacity);

    LLAMA_API void llama_adapter_training_free(struct llama_adapter_training * training);

    // Charged workspace bytes. This is not resident memory or process commit; mapped base pages are excluded.
    LLAMA_API size_t llama_adapter_training_workspace_size(const struct llama_adapter_training * training);

    // Maximum layer-boundary payload bytes, in RAM or the external store; zero for the full reference.
    // External store metadata, alignment and atomic-write space must be reserved separately by the caller.
    LLAMA_API size_t llama_adapter_training_checkpoint_size(const struct llama_adapter_training * training);

    // Compute mean assistant-token loss and fresh adapter gradients without changing any weights.
    // Both arrays have n_tokens elements. labels[i] is the target predicted after tokens[i]; -1 masks that position.
    // Rejects invalid tokens, labels or an entirely masked sequence. Abort/failure invalidates all previous gradients.
    // Repeated calls start from zero recurrent state. Cancellation is observed between GGML nodes.
    LLAMA_API bool llama_adapter_training_compute(
            struct llama_adapter_training * training,
            const llama_token * tokens, const llama_token * labels, float * loss,
            ggml_abort_callback abort_callback, void * abort_data);

    // Read the last successful gradient using the full adapter tensor name, including .lora_a or .lora_b.
    // Returns its element count, or zero if no matching tensor or current successful compute exists.
    // An optimizer step or checkpoint restore invalidates gradients computed against older parameter values.
    // A null or undersized destination is not written; it can be used to query the required element count.
    LLAMA_API size_t llama_adapter_training_gradient(
            const struct llama_adapter_training * training,
            const char * tensor_name, float * data, size_t capacity);

    // Error from the most recent compute call, or an empty string. The pointer lives until the next call or destruction.
    LLAMA_API const char * llama_adapter_training_error(const struct llama_adapter_training * training);

    struct llama_adapter_optimizer_params {
        float beta1;
        float beta2;
        float epsilon;
        float weight_decay;
        float gradient_clip;       // Positive global L2 norm bound, applied after token-weighted accumulation.
        size_t state_budget_bytes; // Three F32 arrays per adapter tensor: first/second moments and gradient sum; excludes container overhead.
    };

    LLAMA_API struct llama_adapter_optimizer_params llama_adapter_optimizer_default_params(void);

    // The caller supplies a SHA-256 binding to its immutable model, tokenizer, dataset and training configuration manifest.
    // Optimizer checkpoint restore requires the same binding, tensor identities, alpha and numerical settings.
    // Owns adapter-only state. The adapter/model are borrowed and require exclusive caller-managed access.
    LLAMA_API struct llama_adapter_optimizer * llama_adapter_optimizer_init(
            struct llama_adapter_lora * adapter, struct llama_adapter_optimizer_params params,
            const char * binding_sha256, char * error, size_t error_capacity);

    LLAMA_API void llama_adapter_optimizer_free(struct llama_adapter_optimizer * optimizer);

    // Consume the last successful gradient once, weighted by its supervised-token count.
    // Different sequence lengths may use different training contexts for the same adapter.
    LLAMA_API bool llama_adapter_optimizer_accumulate(struct llama_adapter_optimizer * optimizer, struct llama_adapter_training * training);

    // Commit one AdamW step and clear the accumulated gradient. Learning-rate scheduling is caller-owned.
    // Validates all candidate values before changing any parameter/state. Failure preserves the uncommitted accumulation.
    LLAMA_API bool llama_adapter_optimizer_step(struct llama_adapter_optimizer * optimizer, float learning_rate);

    LLAMA_API uint64_t llama_adapter_optimizer_step_count(const struct llama_adapter_optimizer * optimizer);
    LLAMA_API uint64_t llama_adapter_optimizer_pending_tokens(const struct llama_adapter_optimizer * optimizer);
    LLAMA_API const char * llama_adapter_optimizer_error(const struct llama_adapter_optimizer * optimizer);

    // GGUF checkpoint includes exact F32 adapter values, AdamW moments, pending gradient sums and step/token counters.
    // Caller owns atomic publication, artifact hashing, dataset position and RNG state in the bound job record.
    // Checkpoint buffers must not alias the adapter or optimizer state. Metadata and output storage are outside the state budget.
    // A null or undersized destination is not written. Returns required bytes, or zero on error.
    LLAMA_API size_t llama_adapter_optimizer_save(struct llama_adapter_optimizer * optimizer, void * data, size_t capacity);

    // Validate the entire checkpoint before committing any state. Rejects mismatched bindings/settings and nonfinite payloads.
    LLAMA_API bool llama_adapter_optimizer_load(struct llama_adapter_optimizer * optimizer, const void * data, size_t size);

#ifdef __cplusplus
}
#endif
