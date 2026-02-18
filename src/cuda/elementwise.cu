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

} // namespace cuda
} // namespace nt
