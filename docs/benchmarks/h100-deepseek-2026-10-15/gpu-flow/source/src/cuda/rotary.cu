#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace nt {
namespace cuda {

// ============================================================
// RoPE (Rotary Position Embedding) kernel
// Supports both interleaved (GPT-NeoX style) and non-interleaved (Llama style)
// ============================================================

// Llama-style RoPE: pairs are (i, i+d/2) not (2i, 2i+1)
__global__ void rope_kernel(
    float* __restrict__ q,       // [batch, seq_len, n_heads, head_dim]
    float* __restrict__ k,       // [batch, seq_len, n_kv_heads, head_dim]
    const int* __restrict__ positions,  // [batch, seq_len]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta_base,
    float freq_scale
) {
    // Each thread handles one pair of elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q = seq_len * n_heads * half_dim;
    int total_k = seq_len * n_kv_heads * half_dim;

    if (idx < total_q + total_k) {
        bool is_key = (idx >= total_q);
        int local_idx = is_key ? (idx - total_q) : idx;

        int n_h = is_key ? n_kv_heads : n_heads;
        int pair_idx = local_idx % half_dim;
        int head = (local_idx / half_dim) % n_h;
        int seq_pos = local_idx / (half_dim * n_h);

        // Get position
        int pos = positions[seq_pos];

        // Compute frequency
        float freq = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
        float angle = pos * freq * freq_scale;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Apply rotation to the pair (i, i + half_dim)
        float* data = is_key ? k : q;
        int stride = n_h * head_dim;
        int base = seq_pos * stride + head * head_dim;

        float x0 = data[base + pair_idx];
        float x1 = data[base + pair_idx + half_dim];

        data[base + pair_idx]            = x0 * cos_val - x1 * sin_val;
        data[base + pair_idx + half_dim] = x1 * cos_val + x0 * sin_val;
    }
}

// Interleaved RoPE: pairs are (2i, 2i+1)
__global__ void rope_interleaved_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta_base,
    float freq_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q = seq_len * n_heads * half_dim;
    int total_k = seq_len * n_kv_heads * half_dim;

    if (idx < total_q + total_k) {
        bool is_key = (idx >= total_q);
        int local_idx = is_key ? (idx - total_q) : idx;

        int n_h = is_key ? n_kv_heads : n_heads;
        int pair_idx = local_idx % half_dim;
        int head = (local_idx / half_dim) % n_h;
        int seq_pos = local_idx / (half_dim * n_h);

        int pos = positions[seq_pos];

        float freq = 1.0f / powf(theta_base, (2.0f * pair_idx) / head_dim);
        float angle = pos * freq * freq_scale;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float* data = is_key ? k : q;
        int stride = n_h * head_dim;
        int base = seq_pos * stride + head * head_dim;

        float x0 = data[base + 2 * pair_idx];
        float x1 = data[base + 2 * pair_idx + 1];

        data[base + 2 * pair_idx]     = x0 * cos_val - x1 * sin_val;
        data[base + 2 * pair_idx + 1] = x1 * cos_val + x0 * sin_val;
    }
}

// ============================================================
// Launcher
// ============================================================

void launch_rope(
    float* q,
    float* k,
    const int* positions,
    int batch_size,   // currently unused, assumed 1 for simplicity
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float theta_base,
    float freq_scale,
    bool interleaved,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int half_dim = head_dim / 2;
    int total = seq_len * (n_heads + n_kv_heads) * half_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    if (interleaved) {
        rope_interleaved_kernel<<<grid, block, 0, s>>>(
            q, k, positions, seq_len, n_heads, n_kv_heads, head_dim, theta_base, freq_scale);
    } else {
        rope_kernel<<<grid, block, 0, s>>>(
            q, k, positions, seq_len, n_heads, n_kv_heads, head_dim, theta_base, freq_scale);
    }
}

} // namespace cuda
} // namespace nt
