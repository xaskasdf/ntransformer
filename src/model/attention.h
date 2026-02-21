#pragma once

#include "../core/tensor.h"
#include "config.h"

namespace nt {

// ============================================================
// Multi-Head Attention with GQA support
// Weights: Wq, Wk, Wv, Wo (all potentially quantized)
// ============================================================

class Attention {
public:
    void init(
        const ModelConfig& config,
        Tensor wq, Tensor wk, Tensor wv, Tensor wo,
        int layer_idx
    );

    // Streaming mode: store config only, skip GPU weight transfer
    void init_streaming(const ModelConfig& config, int layer_idx);

    // Set weights to point at GPU buffer (non-owning views)
    void set_weights(const void* wq, const void* wk, const void* wv, const void* wo,
                     DType wq_dt, DType wk_dt, DType wv_dt, DType wo_dt);

    // Forward pass
    // input:  [seq_len, hidden_size] on GPU
    // output: [seq_len, hidden_size] on GPU
    // For decode: seq_len=1, uses KV cache
    // For prefill: seq_len>1, fills KV cache
    void forward(
        float* output,
        const float* input,
        int seq_len,
        int start_pos,             // position in the KV cache
        void* k_cache,             // [max_seq, n_kv_heads, head_dim] F16
        void* v_cache,             // [max_seq, n_kv_heads, head_dim] F16
        const int* positions,      // [seq_len] position IDs
        void* stream
    );

    // Workspace size needed
    size_t workspace_size(int seq_len) const;

    // Set workspace pointer
    void set_workspace(float* ptr) { workspace_ = ptr; }

private:
    // Quantized weight tensors on GPU
    Tensor wq_, wk_, wv_, wo_;

    // Config
    int hidden_size_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_ = 0;
    float rope_theta_ = 10000.0f;
    float rope_freq_scale_ = 1.0f;
    bool rope_interleaved_ = false;
    float scale_ = 0.0f;
    int layer_idx_ = 0;

    DType wq_dtype_, wk_dtype_, wv_dtype_, wo_dtype_;

    // Workspace pointer (allocated externally)
    float* workspace_ = nullptr;
};

} // namespace nt
