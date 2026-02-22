#include "attention.h"
#include "../cuda/kernels.h"
#include <cmath>
#include <cstdio>

namespace nt {

void Attention::init(
    const ModelConfig& config,
    Tensor wq, Tensor wk, Tensor wv, Tensor wo,
    int layer_idx
) {
    hidden_size_      = config.hidden_size;
    n_heads_          = config.n_heads;
    n_kv_heads_       = config.n_kv_heads;
    head_dim_         = config.head_dim;
    max_seq_          = config.max_seq_len;
    rope_theta_       = config.rope_theta;
    rope_freq_scale_  = config.rope_freq_scale;
    rope_interleaved_ = config.rope_interleaved;
    scale_            = 1.0f / sqrtf((float)head_dim_);
    layer_idx_        = layer_idx;

    wq_dtype_ = wq.dtype();
    wk_dtype_ = wk.dtype();
    wv_dtype_ = wv.dtype();
    wo_dtype_ = wo.dtype();

    // Transfer weights to GPU
    if (wq.device() == Device::CPU) {
        wq_ = wq.to(Device::CUDA);
        wk_ = wk.to(Device::CUDA);
        wv_ = wv.to(Device::CUDA);
        wo_ = wo.to(Device::CUDA);
    } else {
        wq_ = std::move(wq);
        wk_ = std::move(wk);
        wv_ = std::move(wv);
        wo_ = std::move(wo);
    }
}

void Attention::init_streaming(const ModelConfig& config, int layer_idx) {
    hidden_size_      = config.hidden_size;
    n_heads_          = config.n_heads;
    n_kv_heads_       = config.n_kv_heads;
    head_dim_         = config.head_dim;
    max_seq_          = config.max_seq_len;
    rope_theta_       = config.rope_theta;
    rope_freq_scale_  = config.rope_freq_scale;
    rope_interleaved_ = config.rope_interleaved;
    scale_            = 1.0f / sqrtf((float)head_dim_);
    layer_idx_        = layer_idx;
}

void Attention::set_base_weights(const void* base_q, const void* base_k,
                                 const void* base_v, const void* base_o,
                                 DType base_dtype) {
    base_wq_ = base_q;
    base_wk_ = base_k;
    base_wv_ = base_v;
    base_wo_ = base_o;
    base_dtype_ = base_dtype;
}

void Attention::set_delta(const void* uq, const void* vq,
                          const void* uk, const void* vk,
                          const void* uv, const void* vv,
                          const void* uo, const void* vo,
                          int rank, float* temp_buf) {
    delta_mode_ = true;
    delta_rank_ = rank;
    delta_uq_ = uq; delta_vq_ = vq;
    delta_uk_ = uk; delta_vk_ = vk;
    delta_uv_ = uv; delta_vv_ = vv;
    delta_uo_ = uo; delta_vo_ = vo;
    delta_temp_ = temp_buf;
}

void Attention::set_weights(const void* wq, const void* wk, const void* wv, const void* wo,
                            DType wq_dt, DType wk_dt, DType wv_dt, DType wo_dt) {
    delta_mode_ = false;  // Explicit weights disable delta mode
    wq_dtype_ = wq_dt;
    wk_dtype_ = wk_dt;
    wv_dtype_ = wv_dt;
    wo_dtype_ = wo_dt;

    int q_dim = n_heads_ * head_dim_;
    int kv_dim = n_kv_heads_ * head_dim_;

    wq_ = Tensor::from_ptr(const_cast<void*>(wq), {q_dim, hidden_size_}, wq_dt, Device::CUDA);
    wk_ = Tensor::from_ptr(const_cast<void*>(wk), {kv_dim, hidden_size_}, wk_dt, Device::CUDA);
    wv_ = Tensor::from_ptr(const_cast<void*>(wv), {kv_dim, hidden_size_}, wv_dt, Device::CUDA);
    wo_ = Tensor::from_ptr(const_cast<void*>(wo), {hidden_size_, q_dim}, wo_dt, Device::CUDA);
}

// Delta GEMV: y = Base*x + U*(V^T*x)
static void delta_gemv(float* y, const void* base_W, DType base_dt,
                       const void* U, const void* V, float* temp_r,
                       const float* x, int out, int in, int rank, void* stream) {
    cuda::launch_gemv(y, base_W, x, out, in, base_dt, stream);             // y = Base*x
    cuda::launch_gemv(temp_r, V, x, rank, in, DType::F16, stream);         // temp = V^T*x
    cuda::launch_gemv_add(y, U, temp_r, out, rank, DType::F16, stream);    // y += U*temp
}

size_t Attention::workspace_size(int seq_len) const {
    // Need space for:
    // q_buf: [seq_len, n_heads * head_dim]
    // k_buf: [seq_len, n_kv_heads * head_dim]
    // v_buf: [seq_len, n_kv_heads * head_dim]
    // attn_out: [seq_len, n_heads * head_dim]
    // positions_gpu: [seq_len] ints
    size_t q_size = seq_len * n_heads_ * head_dim_;
    size_t k_size = seq_len * n_kv_heads_ * head_dim_;
    size_t v_size = seq_len * n_kv_heads_ * head_dim_;
    size_t out_size = seq_len * n_heads_ * head_dim_;
    return (q_size + k_size + v_size + out_size) * sizeof(float);
}

void Attention::forward(
    float* output,
    const float* input,
    int seq_len,
    int start_pos,
    void* k_cache,
    void* v_cache,
    const int* positions,
    void* stream
) {
    NT_CHECK(workspace_ != nullptr, "Attention workspace not set");

    // Workspace layout
    float* q_buf = workspace_;
    float* k_buf = q_buf + seq_len * n_heads_ * head_dim_;
    float* v_buf = k_buf + seq_len * n_kv_heads_ * head_dim_;
    float* attn_out = v_buf + seq_len * n_kv_heads_ * head_dim_;

    // Project Q, K, V
    // For single token decode, these are GEMV operations
    // For prefill, they'd be GEMM (we use GEMV per-token for now)
    int q_dim = n_heads_ * head_dim_;
    int kv_dim = n_kv_heads_ * head_dim_;

    for (int t = 0; t < seq_len; t++) {
        const float* inp = input + t * hidden_size_;
        float* q_out = q_buf + t * q_dim;
        float* k_out = k_buf + t * kv_dim;
        float* v_out = v_buf + t * kv_dim;

        if (delta_mode_) {
            delta_gemv(q_out, base_wq_, base_dtype_, delta_uq_, delta_vq_,
                       delta_temp_, inp, q_dim, hidden_size_, delta_rank_, stream);
            delta_gemv(k_out, base_wk_, base_dtype_, delta_uk_, delta_vk_,
                       delta_temp_, inp, kv_dim, hidden_size_, delta_rank_, stream);
            delta_gemv(v_out, base_wv_, base_dtype_, delta_uv_, delta_vv_,
                       delta_temp_, inp, kv_dim, hidden_size_, delta_rank_, stream);
        } else {
            cuda::launch_gemv(q_out, wq_.data(), inp, q_dim, hidden_size_, wq_dtype_, stream);
            cuda::launch_gemv(k_out, wk_.data(), inp, kv_dim, hidden_size_, wk_dtype_, stream);
            cuda::launch_gemv(v_out, wv_.data(), inp, kv_dim, hidden_size_, wv_dtype_, stream);
        }
    }

    // Apply RoPE to Q and K
    cuda::launch_rope(
        q_buf, k_buf, positions,
        1, seq_len, n_heads_, n_kv_heads_, head_dim_,
        rope_theta_, rope_freq_scale_, rope_interleaved_,
        stream
    );

    // Store K, V in cache
    cuda::launch_copy_to_kv_cache(
        k_cache, v_cache, k_buf, v_buf,
        seq_len, n_kv_heads_, head_dim_, start_pos, max_seq_,
        stream
    );

    // Compute attention
    int total_seq = start_pos + seq_len;

    if (seq_len == 1) {
        // Decode: single query against full cache
        cuda::launch_attention_decode(
            attn_out, q_buf, k_cache, v_cache,
            total_seq, n_heads_, n_kv_heads_, head_dim_, max_seq_,
            scale_, stream
        );
    } else {
        // Prefill / multi-token decode: batch of queries with causal mask
        // Reads K,V from cache (already stored by copy_to_kv_cache above)
        cuda::launch_attention_prefill(
            attn_out, q_buf, k_cache, v_cache,
            seq_len, start_pos, n_heads_, n_kv_heads_, head_dim_, max_seq_,
            scale_, stream
        );
    }

    // Output projection: attn_out -> output via Wo
    for (int t = 0; t < seq_len; t++) {
        float* a_out = attn_out + t * q_dim;
        float* out = output + t * hidden_size_;
        if (delta_mode_) {
            delta_gemv(out, base_wo_, base_dtype_, delta_uo_, delta_vo_,
                       delta_temp_, a_out, hidden_size_, q_dim, delta_rank_, stream);
        } else {
            cuda::launch_gemv(out, wo_.data(), a_out,
                hidden_size_, q_dim, wo_dtype_, stream);
        }
    }
}

} // namespace nt
