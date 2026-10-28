#include "ds41_attention.h"
#include <cmath>

namespace nt::ds41::detail {
namespace {
__device__ const float* key_row(const float* raw, int window, int64_t first, int n_raw,
                               const float* compressed, const int* selected, int j, int dim) {
    return j < n_raw ? raw + (size_t)((first + j) % window) * dim
                    : compressed + (size_t)selected[j - n_raw] * dim;
}

// Each of sixteen lanes follows a whole, ordered FP32 dot product. Other
// threads cooperatively load keys, avoiding dependent strided global reads.
__global__ void attention_scores(const float* query, const float* raw, int window,
                                 int64_t first, int n_raw, const float* compressed,
                                 const int* selected, int n_keys, int dim, float scale, float* scores) {
    extern __shared__ float storage[];
    float* q = storage;
    float* keys = storage + dim;
    const int head = blockIdx.x, key0 = blockIdx.y * 16;
    const int count = min(16, n_keys - key0);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) q[d] = query[(size_t)head * dim + d];
    for (int i = threadIdx.x; i < count * dim; i += blockDim.x) {
        const int k = i / dim, d = i % dim;
        const float* row = key_row(raw, window, first, n_raw, compressed, selected, key0 + k, dim);
        // Odd pitch avoids bank conflicts in the cooperative transpose.
        keys[d * 17 + k] = row[d];
    }
    __syncthreads();
    if (threadIdx.x < count) {
        float dot = 0;
        for (int d = 0; d < dim; ++d)
            dot = __fadd_rn(dot, __fmul_rn(q[d], keys[d * 17 + threadIdx.x]));
        scores[(size_t)head * n_keys + key0 + threadIdx.x] = __fmul_rn(dot, scale);
    }
}

__global__ void attention_values(const float* scores, const float* raw, int window,
                                 int64_t first, int n_raw, const float* compressed,
                                 const int* selected, int n_keys, const float* sinks,
                                 const float* cos_t, const float* sin_t,
                                 int dim, int rope_dim, float* out) {
    extern __shared__ float probabilities[];
    __shared__ float maximum, inverse;
    const int head = blockIdx.x;
    const float* sc = scores + (size_t)head * n_keys;
    if (threadIdx.x == 0) {
        float mx = sinks[head];
        for (int j = 0; j < n_keys; ++j) if (sc[j] > mx) mx = sc[j];
        maximum = mx;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x)
        probabilities[j] = (float)exp((double)__fsub_rn(sc[j], maximum));
    __syncthreads();
    if (threadIdx.x == 0) {
        float denom = (float)exp((double)__fsub_rn(sinks[head], maximum));
        for (int j = 0; j < n_keys; ++j) denom = __fadd_rn(denom, probabilities[j]);
        inverse = __fdiv_rn(1.0f, denom);
    }
    __syncthreads();
    float* result = out + (size_t)head * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0;
        for (int j = 0; j < n_keys; ++j) {
            const float* row = key_row(raw, window, first, n_raw, compressed, selected, j, dim);
            acc = __fmaf_rn(probabilities[j], row[d], acc);
        }
        result[d] = __fmul_rn(acc, inverse);
    }
    __syncthreads();
    // Consecutive RoPE pairs, inverse rotation at the current position.
    for (int p = threadIdx.x; p < rope_dim / 2; p += blockDim.x) {
        const int d = dim - rope_dim + 2 * p;
        const float a = result[d], b = result[d + 1], c = cos_t[p], s = -sin_t[p];
        result[d] = __fmaf_rn(a, c, -__fmul_rn(b, s));
        result[d + 1] = __fmaf_rn(a, s, __fmul_rn(b, c));
    }
}
}

void launch_attention(const float* query, const float* raw, int window, int64_t position,
                      const float* compressed, const int* selected, int n_selected,
                      const float* sinks, const float* cos_t, const float* sin_t,
                      int heads, int dim, int rope_dim, float scale,
                      float* scores, float* out, cudaStream_t stream) {
    const int n_raw = (int)min((int64_t)window, position + 1);
    const int64_t first = position + 1 - n_raw;
    const int n_keys = n_raw + n_selected;
    attention_scores<<<dim3(heads, (n_keys + 15) / 16), 256, (size_t)dim * 18 * 4, stream>>>(
        query, raw, window, first, n_raw, compressed, selected, n_keys, dim, scale, scores);
    attention_values<<<heads, 256, (size_t)n_keys * 4, stream>>>(
        scores, raw, window, first, n_raw, compressed, selected, n_keys, sinks,
        cos_t, sin_t, dim, rope_dim, out);
}
}
