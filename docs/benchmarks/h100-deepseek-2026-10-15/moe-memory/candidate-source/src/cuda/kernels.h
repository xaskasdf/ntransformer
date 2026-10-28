#pragma once

#include "../core/types.h"

// ============================================================
// CUDA kernel launcher declarations
// All functions take a void* stream (cast to cudaStream_t internally)
// ============================================================

namespace nt {
namespace cuda {

// RMSNorm
void launch_rmsnorm(
    float* output, const float* input, const float* weight,
    int batch_size, int hidden_size, float eps, void* stream);
void launch_rmsnorm_f16(
    void* output, const float* input, const float* weight,
    int batch_size, int hidden_size, float eps, void* stream);

// RoPE
void launch_rope(
    float* q, float* k, const int* positions,
    int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
    float theta_base, float freq_scale, bool interleaved, void* stream);

// Softmax
void launch_softmax(float* output, const float* input, int rows, int cols, void* stream);
void launch_masked_softmax(
    float* output, const float* input, const bool* mask,
    int rows, int cols, void* stream);

// GEMV/GEMM
void launch_gemv(
    float* y, const void* W, const float* x,
    int out_features, int in_features, DType weight_dtype, void* stream);

// Accumulate GEMV: y += W * x (F16 weights only, for delta encoding)
void launch_gemv_add(
    float* y, const void* W, const float* x,
    int out_features, int in_features, DType weight_dtype, void* stream);
void launch_gemm_f32(
    float* C, const float* A, const float* B,
    int M, int N, int K, void* stream);
void launch_silu_mul(
    float* output, const float* gate, const float* up,
    int size, void* stream);
void launch_add_bias(float* y, const float* bias, int size, void* stream);

// Attention (KV cache is F16 — void* to avoid half/uint16_t ABI mismatch)
void launch_attention_decode(
    float* output, const float* q, const void* k_cache, const void* v_cache,
    int seq_len, int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale, void* stream);
void launch_attention_prefill(
    float* output, const float* Q,
    const void* k_cache, const void* v_cache,
    int seq_len, int start_pos, int n_heads, int n_kv_heads,
    int head_dim, int max_seq, float scale, void* stream);
void launch_copy_to_kv_cache(
    void* k_cache, void* v_cache, const float* k, const float* v,
    int seq_len, int n_kv_heads, int head_dim, int start_pos, int max_seq,
    void* stream);

// Element-wise ops
void launch_add(float* out, const float* a, const float* b, int size, void* stream);
void launch_add_inplace(float* a, const float* b, int size, void* stream);
void launch_copy(float* dst, const float* src, int size, void* stream);

// Cosine similarity (single-block reduction, writes result to device float)
void launch_cosine_similarity(float* result, const float* a, const float* b, int size, void* stream);

} // namespace cuda
} // namespace nt
