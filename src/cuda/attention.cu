#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace nt {
namespace cuda {

// ============================================================
// Flash Attention decode kernel (single query token)
// Handles GQA (Grouped Query Attention) natively
//
// For decode: q is [n_heads, head_dim], k/v_cache are [seq_len, n_kv_heads, head_dim]
// GQA: n_heads / n_kv_heads = group_size (each KV head shared by group_size Q heads)
//
// Each block handles one query head.
// Uses online softmax for O(1) extra memory per head.
// ============================================================

template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_decode_kernel(
    float* __restrict__ output,          // [n_heads, head_dim]
    const float* __restrict__ q,         // [n_heads, head_dim]
    const float* __restrict__ k_cache,   // [max_seq, n_kv_heads, head_dim]
    const float* __restrict__ v_cache,   // [max_seq, n_kv_heads, head_dim]
    int seq_len,                          // current sequence length
    int n_heads,
    int n_kv_heads,
    int max_seq,
    float scale
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int kv_head = head / (n_heads / n_kv_heads);  // GQA mapping

    const float* q_head = q + head * HEAD_DIM;

    // Shared memory for q vector (loaded once)
    __shared__ float s_q[HEAD_DIM];
    if (tid < HEAD_DIM) {
        s_q[tid] = q_head[tid];
    }
    __syncthreads();

    // Online softmax accumulators
    float m_prev = -FLT_MAX;  // running max
    float l_prev = 0.0f;       // running sum of exp
    float acc[HEAD_DIM / 32];  // partial output accumulator (HEAD_DIM/BLOCK_SIZE elements per thread)

    // Initialize accumulator
    constexpr int ELEMS_PER_THREAD = (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float local_acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        local_acc[i] = 0.0f;
    }

    // Iterate over KV cache positions
    for (int pos = tid; pos < seq_len; pos += BLOCK_SIZE) {
        const float* k_pos = k_cache + pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
        const float* v_pos = v_cache + pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;

        // Compute attention score: dot(q, k) * scale
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            score += s_q[d] * k_pos[d];
        }
        score *= scale;

        // Online softmax update
        float m_new = fmaxf(m_prev, score);
        float exp_prev = expf(m_prev - m_new);
        float exp_curr = expf(score - m_new);
        float l_new = l_prev * exp_prev + exp_curr;

        // Update accumulator: rescale old + add new
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int d = i * BLOCK_SIZE + tid;
            if (d < HEAD_DIM) {
                local_acc[i] = local_acc[i] * (l_prev * exp_prev / fmaxf(l_new, 1e-10f))
                             + (exp_curr / fmaxf(l_new, 1e-10f)) * v_pos[d];
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output - each thread writes its elements
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int d = i * BLOCK_SIZE + tid;
        if (d < HEAD_DIM) {
            output[head * HEAD_DIM + d] = local_acc[i];
        }
    }
}

// ============================================================
// Simpler decode attention for arbitrary head_dim
// One block per head, uses shared memory for reduction
// ============================================================

__global__ void attention_decode_generic_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int kv_head = head / (n_heads / n_kv_heads);

    const float* q_head = q + head * head_dim;

    // Shared memory: scores for all positions, then for reduction
    extern __shared__ float smem[];  // [seq_len] for scores

    // Phase 1: compute all attention scores
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float* k_pos = k_cache + pos * n_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_head[d] * k_pos[d];
        }
        smem[pos] = score * scale;
    }
    __syncthreads();

    // Phase 2: softmax over scores (single thread for simplicity, or parallel)
    // Find max
    float local_max = -FLT_MAX;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, smem[pos]);
    }
    // Warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    __shared__ float shared_vals[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) shared_vals[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (tid < blockDim.x / warpSize) ? shared_vals[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
    }
    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float val = expf(smem[pos] - s_max);
        smem[pos] = val;
        local_sum += val;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) shared_vals[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        local_sum = (tid < blockDim.x / warpSize) ? shared_vals[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
    }
    __shared__ float s_sum_inv;
    if (tid == 0) s_sum_inv = 1.0f / local_sum;
    __syncthreads();

    // Normalize
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        smem[pos] *= s_sum_inv;
    }
    __syncthreads();

    // Phase 3: weighted sum of values
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            const float* v_pos = v_cache + pos * n_kv_heads * head_dim + kv_head * head_dim;
            sum += smem[pos] * v_pos[d];
        }
        output[head * head_dim + d] = sum;
    }
}

// ============================================================
// Prefill attention (batch of queries, reads K/V from cache)
// Q: [seq_len, n_heads, head_dim] (batch buffer)
// k_cache/v_cache: [max_seq, n_kv_heads, head_dim] (full KV cache)
// Output: [seq_len, n_heads, head_dim]
// With causal mask based on absolute positions
//
// Each block handles one (head, batch_position) pair.
// For query at batch position q_idx (absolute position start_pos + q_idx),
// attends to all cache positions 0..start_pos+q_idx with causal mask.
// ============================================================

__global__ void attention_prefill_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,           // [seq_len, n_heads, head_dim]
    const float* __restrict__ k_cache,     // [max_seq, n_kv_heads, head_dim]
    const float* __restrict__ v_cache,     // [max_seq, n_kv_heads, head_dim]
    int seq_len,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float scale
) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y;            // 0..seq_len-1 (batch index)
    int q_abs = start_pos + q_idx;     // absolute position in sequence
    int tid = threadIdx.x;
    int kv_head = head / (n_heads / n_kv_heads);

    const float* q_vec = Q + q_idx * n_heads * head_dim + head * head_dim;

    extern __shared__ float smem[];
    // smem layout: [q_abs + 1] for scores (all positions 0..q_abs)

    int n_keys = q_abs + 1;  // attend to absolute positions 0..q_abs

    // Compute scores: Q dot K for all positions 0..q_abs from cache
    for (int k_pos = tid; k_pos < n_keys; k_pos += blockDim.x) {
        const float* k_vec = k_cache + k_pos * n_kv_heads * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * k_vec[d];
        }
        smem[k_pos] = score * scale;
    }
    __syncthreads();

    // Softmax over [0..q_abs]
    float local_max = -FLT_MAX;
    for (int pos = tid; pos < n_keys; pos += blockDim.x) {
        local_max = fmaxf(local_max, smem[pos]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    __shared__ float sv[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) sv[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (tid < blockDim.x / warpSize) ? sv[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
    }
    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int pos = tid; pos < n_keys; pos += blockDim.x) {
        float val = expf(smem[pos] - s_max);
        smem[pos] = val;
        local_sum += val;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) sv[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        local_sum = (tid < blockDim.x / warpSize) ? sv[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
    }
    __shared__ float s_sum_inv;
    if (tid == 0) s_sum_inv = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;
    __syncthreads();

    for (int pos = tid; pos < n_keys; pos += blockDim.x) {
        smem[pos] *= s_sum_inv;
    }
    __syncthreads();

    // Weighted sum of values from cache
    float* out_vec = output + q_idx * n_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int pos = 0; pos < n_keys; pos++) {
            const float* v_vec = v_cache + pos * n_kv_heads * head_dim + kv_head * head_dim;
            sum += smem[pos] * v_vec[d];
        }
        out_vec[d] = sum;
    }
}

// ============================================================
// Copy Q/K/V to cache kernel
// ============================================================
__global__ void copy_to_kv_cache_kernel(
    float* __restrict__ k_cache,  // [max_seq, n_kv_heads, head_dim]
    float* __restrict__ v_cache,
    const float* __restrict__ k,  // [seq_len, n_kv_heads, head_dim]
    const float* __restrict__ v,
    int seq_len,
    int n_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_kv_heads * head_dim;

    if (idx < total) {
        int d = idx % head_dim;
        int kv_head = (idx / head_dim) % n_kv_heads;
        int seq = idx / (head_dim * n_kv_heads);

        int cache_pos = start_pos + seq;
        if (cache_pos < max_seq) {
            int cache_idx = cache_pos * n_kv_heads * head_dim + kv_head * head_dim + d;
            k_cache[cache_idx] = k[idx];
            v_cache[cache_idx] = v[idx];
        }
    }
}

// ============================================================
// Launchers
// ============================================================

void launch_attention_decode(
    float* output,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // Use generic kernel - works for any head_dim
    int block_size = 256;
    if (seq_len > 1024) block_size = 512;

    size_t smem_size = seq_len * sizeof(float);
    dim3 grid(n_heads);

    attention_decode_generic_kernel<<<grid, block_size, smem_size, s>>>(
        output, q, k_cache, v_cache, seq_len, n_heads, n_kv_heads, head_dim, max_seq, scale);
}

void launch_attention_prefill(
    float* output,
    const float* Q,
    const float* k_cache,
    const float* v_cache,
    int seq_len,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    int block_size = 256;
    int total_seq = start_pos + seq_len;
    size_t smem_size = total_seq * sizeof(float);  // worst case: last position needs all scores
    dim3 grid(n_heads, seq_len);

    attention_prefill_kernel<<<grid, block_size, smem_size, s>>>(
        output, Q, k_cache, v_cache, seq_len, start_pos,
        n_heads, n_kv_heads, head_dim, scale);
}

void launch_copy_to_kv_cache(
    float* k_cache,
    float* v_cache,
    const float* k,
    const float* v,
    int seq_len,
    int n_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int total = seq_len * n_kv_heads * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    copy_to_kv_cache_kernel<<<grid, block, 0, s>>>(
        k_cache, v_cache, k, v, seq_len, n_kv_heads, head_dim, start_pos, max_seq);
}

} // namespace cuda
} // namespace nt
