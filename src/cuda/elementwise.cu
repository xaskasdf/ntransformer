#include "../core/types.h"
#include <cuda_runtime.h>

namespace nt {
namespace cuda {

// ============================================================
// Element-wise operations
// ============================================================

__global__ void add_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

__global__ void add_inplace_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

__global__ void copy_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = src[i];
    }
}

// Cosine similarity: cos(a, b) = dot(a,b) / (||a|| * ||b||)
// Single block, 256 threads, works for size up to ~65K
__global__ void cosine_similarity_kernel(
    float* __restrict__ result,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int size
) {
    __shared__ float s_dot[256];
    __shared__ float s_na[256];
    __shared__ float s_nb[256];

    int tid = threadIdx.x;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float ai = a[i], bi = b[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }

    s_dot[tid] = dot;
    s_na[tid] = na;
    s_nb[tid] = nb;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_na[tid] += s_na[tid + s];
            s_nb[tid] += s_nb[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = sqrtf(s_na[0]) * sqrtf(s_nb[0]);
        *result = (denom > 1e-8f) ? s_dot[0] / denom : 0.0f;
    }
}

// ============================================================
// Launchers
// ============================================================

void launch_add(float* out, const float* a, const float* b, int size, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid = (size + block - 1) / block;
    add_kernel<<<grid, block, 0, s>>>(out, a, b, size);
}

void launch_add_inplace(float* a, const float* b, int size, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid = (size + block - 1) / block;
    add_inplace_kernel<<<grid, block, 0, s>>>(a, b, size);
}

void launch_copy(float* dst, const float* src, int size, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid = (size + block - 1) / block;
    copy_kernel<<<grid, block, 0, s>>>(dst, src, size);
}

void launch_cosine_similarity(float* result, const float* a, const float* b, int size, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cosine_similarity_kernel<<<1, 256, 0, s>>>(result, a, b, size);
}

} // namespace cuda
} // namespace nt
