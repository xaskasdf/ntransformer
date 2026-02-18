#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace nt {
namespace cuda {

// ============================================================
// Online softmax kernel
// Numerically stable: max and sum computed in a single pass
// Uses the "online" algorithm from Flash Attention paper
// ============================================================

// Single-row softmax with one block per row
template<int BLOCK_SIZE>
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* x = input + row * cols;
    float* y = output + row * cols;

    // Phase 1: find max (for numerical stability)
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }

    __shared__ float shared_max[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    if (lane == 0) shared_max[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (tid < (BLOCK_SIZE / warpSize)) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
    }

    __shared__ float row_max;
    if (tid == 0) row_max = local_max;
    __syncthreads();
    float max_val = row_max;

    // Phase 2: compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = expf(x[i] - max_val);
        y[i] = val;
        local_sum += val;
    }

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (tid < (BLOCK_SIZE / warpSize)) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
    }

    __shared__ float row_sum_inv;
    if (tid == 0) row_sum_inv = 1.0f / local_sum;
    __syncthreads();
    float sum_inv = row_sum_inv;

    // Phase 3: normalize
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        y[i] *= sum_inv;
    }
}

// Masked softmax for attention (mask value = -inf for invalid positions)
template<int BLOCK_SIZE>
__global__ void masked_softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const bool* __restrict__ mask,  // true = keep, false = -inf
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* x = input + row * cols;
    float* y = output + row * cols;
    const bool* m = mask ? (mask + row * cols) : nullptr;

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = (m == nullptr || m[i]) ? x[i] : -FLT_MAX;
        local_max = fmaxf(local_max, val);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }

    __shared__ float shared_max[32];
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    if (lane == 0) shared_max[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (tid < (BLOCK_SIZE / warpSize)) ? shared_max[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
    }
    __shared__ float row_max;
    if (tid == 0) row_max = local_max;
    __syncthreads();
    float max_val = row_max;

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = (m == nullptr || m[i]) ? expf(x[i] - max_val) : 0.0f;
        y[i] = val;
        local_sum += val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        local_sum = (tid < (BLOCK_SIZE / warpSize)) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
    }
    __shared__ float row_sum_inv;
    if (tid == 0) row_sum_inv = (local_sum > 0.0f) ? (1.0f / local_sum) : 0.0f;
    __syncthreads();
    float sum_inv = row_sum_inv;

    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        y[i] *= sum_inv;
    }
}

// ============================================================
// Launchers
// ============================================================

void launch_softmax(
    float* output,
    const float* input,
    int rows,
    int cols,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(rows);

    if (cols <= 1024) {
        softmax_kernel<256><<<grid, 256, 0, s>>>(output, input, cols);
    } else if (cols <= 4096) {
        softmax_kernel<512><<<grid, 512, 0, s>>>(output, input, cols);
    } else {
        softmax_kernel<1024><<<grid, 1024, 0, s>>>(output, input, cols);
    }
}

void launch_masked_softmax(
    float* output,
    const float* input,
    const bool* mask,
    int rows,
    int cols,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(rows);

    if (cols <= 1024) {
        masked_softmax_kernel<256><<<grid, 256, 0, s>>>(output, input, mask, cols);
    } else if (cols <= 4096) {
        masked_softmax_kernel<512><<<grid, 512, 0, s>>>(output, input, mask, cols);
    } else {
        masked_softmax_kernel<1024><<<grid, 1024, 0, s>>>(output, input, mask, cols);
    }
}

} // namespace cuda
} // namespace nt
