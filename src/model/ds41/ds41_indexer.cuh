#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>

namespace nt::ds41::detail {
namespace {
// Ordered FP32 dots match the scalar reduction used by the CPU indexer.
// Sixteen independent keys share a coalesced load and a broadcast query.
__global__ void indexer_scores(const float* query, const float* keys, int count,
                               int dim, float* scores) {
    extern __shared__ float storage[];
    float* q = storage;
    float* tile = storage + dim;
    const int head = blockIdx.x, first = blockIdx.y * 16;
    const int available = min(16, count - first);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) q[d] = query[(size_t)head * dim + d];
    for (int i = threadIdx.x; i < available * dim; i += blockDim.x) {
        const int row = i / dim, d = i % dim;
        tile[d * 17 + row] = keys[(size_t)(first + row) * dim + d];
    }
    __syncthreads();
    if (threadIdx.x < available) {
        float dot = 0;
        for (int d = 0; d < dim; ++d)
            dot = __fadd_rn(dot, __fmul_rn(q[d], tile[d * 17 + threadIdx.x]));
        scores[(size_t)head * count + first + threadIdx.x] = dot;
    }
}

__device__ bool indexer_before(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}

// A bounded, single-CTA sort keeps scores and selected indices on device.
// Equal scores choose the lower row ID, making the cutoff reproducible.
__global__ void indexer_select(const float* dots, const float* weights, int heads,
                               int count, int padded, int top_k, int* selected) {
    extern __shared__ float storage[];
    float* scores = storage;
    int* ids = reinterpret_cast<int*>(storage + padded);
    for (int j = threadIdx.x; j < padded; j += blockDim.x) {
        float score = 0;
        if (j < count) {
            for (int h = 0; h < heads; ++h) {
                const float dot = dots[(size_t)h * count + j];
                if (dot > 0) score = __fmaf_rn(dot, weights[h], score);
            }
        }
        scores[j] = j < count ? score : -CUDART_INF_F;
        ids[j] = j < count ? j : INT_MAX;
    }
    __syncthreads();
    for (int width = 2; width <= padded; width *= 2) {
        for (int stride = width / 2; stride; stride /= 2) {
            for (int j = threadIdx.x; j < padded; j += blockDim.x) {
                const int other = j ^ stride;
                if (other > j) {
                    const bool descending = (j & width) == 0;
                    const bool swap = descending ? indexer_before(scores[other], ids[other], scores[j], ids[j])
                                                 : indexer_before(scores[j], ids[j], scores[other], ids[other]);
                    if (swap) {
                        const float score = scores[j]; scores[j] = scores[other]; scores[other] = score;
                        const int id = ids[j]; ids[j] = ids[other]; ids[other] = id;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int j = threadIdx.x; j < top_k; j += blockDim.x) selected[j] = ids[j];
}

__global__ void indexer_scale(float* weights, int heads, float scale) {
    const int h = threadIdx.x;
    if (h < heads) weights[h] = __fmul_rn(weights[h], scale);
}
} // namespace

inline void launch_indexer_select(const float* query, const float* keys, const float* weights,
                                  int heads, int dim, int count, int top_k,
                                  float* scores, int* selected, cudaStream_t stream) {
    indexer_scores<<<dim3(heads, (count + 15) / 16), 256, (size_t)dim * 18 * sizeof(float), stream>>>(
        query, keys, count, dim, scores);
    int padded = 1;
    while (padded < count) padded *= 2;
    indexer_select<<<1, 256, (size_t)padded * (sizeof(float) + sizeof(int)), stream>>>(
        scores, weights, heads, count, padded, top_k, selected);
}
} // namespace nt::ds41::detail
