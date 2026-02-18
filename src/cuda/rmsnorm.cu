#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace nt {
namespace cuda {

// ============================================================
// Fused RMSNorm kernel
// y = (x / sqrt(mean(x^2) + eps)) * weight
// Single-pass: compute sum of squares with warp reduction
// ============================================================

// For small hidden dims (<=8192), use a single block per row
template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* x = input + row * hidden_size;
    float* y = output + row * hidden_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        sum_sq = (tid < (BLOCK_SIZE / warpSize)) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    // Broadcast rms_inv to all threads
    __shared__ float rms_inv_shared;
    if (tid == 0) {
        float mean_sq = sum_sq / hidden_size;
        rms_inv_shared = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rms_inv = rms_inv_shared;

    // Apply normalization and scale
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        y[i] = x[i] * rms_inv * weight[i];
    }
}

// Half-precision output variant (for feeding into quantized GEMV)
template<int BLOCK_SIZE>
__global__ void rmsnorm_f16_kernel(
    half* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* x = input + row * hidden_size;
    half* y = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = x[i];
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float shared[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        sum_sq = (tid < (BLOCK_SIZE / warpSize)) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
        }
    }

    __shared__ float rms_inv_shared;
    if (tid == 0) {
        float mean_sq = sum_sq / hidden_size;
        rms_inv_shared = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rms_inv = rms_inv_shared;

    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = x[i] * rms_inv * weight[i];
        y[i] = __float2half(val);
    }
}

// ============================================================
// Launcher
// ============================================================

void launch_rmsnorm(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int hidden_size,
    float eps,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(batch_size);

    if (hidden_size <= 1024) {
        rmsnorm_kernel<256><<<grid, 256, 0, s>>>(output, input, weight, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        rmsnorm_kernel<512><<<grid, 512, 0, s>>>(output, input, weight, hidden_size, eps);
    } else {
        rmsnorm_kernel<1024><<<grid, 1024, 0, s>>>(output, input, weight, hidden_size, eps);
    }
}

void launch_rmsnorm_f16(
    void* output,  // half*
    const float* input,
    const float* weight,
    int batch_size,
    int hidden_size,
    float eps,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(batch_size);
    half* out = static_cast<half*>(output);

    if (hidden_size <= 1024) {
        rmsnorm_f16_kernel<256><<<grid, 256, 0, s>>>(out, input, weight, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        rmsnorm_f16_kernel<512><<<grid, 512, 0, s>>>(out, input, weight, hidden_size, eps);
    } else {
        rmsnorm_f16_kernel<1024><<<grid, 1024, 0, s>>>(out, input, weight, hidden_size, eps);
    }
}

} // namespace cuda
} // namespace nt
