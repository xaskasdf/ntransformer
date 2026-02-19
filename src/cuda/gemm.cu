#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace nt {
namespace cuda {

// ============================================================
// Quantized GEMV (matrix-vector multiply) kernels
// These are the CRITICAL kernels for inference performance.
//
// For decode (single token): y = W * x  where W is quantized
// W: [out_features, in_features] in quantized format
// x: [in_features] in FP32
// y: [out_features] in FP32
//
// Each warp computes one output element
// ============================================================

// ----------------------------------------------------------
// Q4_0 GEMV: 32 weights per block, half scale + 16 nibble bytes
// ----------------------------------------------------------
__global__ void gemv_q4_0_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,  // BlockQ4_0 array
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int tid = threadIdx.x;  // lane within warp
    int num_blocks = in_features / 32;  // Q4_0 blocks per row

    const nt::BlockQ4_0* row_blocks = reinterpret_cast<const nt::BlockQ4_0*>(W) + row * num_blocks;

    float sum = 0.0f;

    // Each thread in warp processes some blocks
    for (int b = tid; b < num_blocks; b += warpSize) {
        const nt::BlockQ4_0& block = row_blocks[b];

        // Decode FP16 scale
        float d = __half2float(*reinterpret_cast<const half*>(&block.d));

        float block_sum = 0.0f;
        int base = b * 32;

        // Unpack 32 nibbles from 16 bytes
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t byte = block.qs[j];
            int8_t lo = (byte & 0x0F) - 8;  // Q4_0: values are 0-15, subtract 8 for zero-point
            int8_t hi = (byte >> 4) - 8;

            block_sum += lo * x[base + j];
            block_sum += hi * x[base + j + 16];
        }

        sum += d * block_sum;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Q8_0 GEMV: 32 weights per block, float scale + 32 int8
// ----------------------------------------------------------
__global__ void gemv_q8_0_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int tid = threadIdx.x;
    int num_blocks = in_features / 32;

    const nt::BlockQ8_0* row_blocks = reinterpret_cast<const nt::BlockQ8_0*>(W) + row * num_blocks;

    float sum = 0.0f;

    for (int b = tid; b < num_blocks; b += warpSize) {
        const nt::BlockQ8_0& block = row_blocks[b];
        // Decode FP16 scale
        float d = __half2float(*reinterpret_cast<const half*>(&block.d));
        int base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            block_sum += block.qs[j] * x[base + j];
        }

        sum += d * block_sum;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Q4_K_M GEMV: 256 weights per super-block
// More complex layout with sub-block scales
// ----------------------------------------------------------
__global__ void gemv_q4_k_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int tid = threadIdx.x;
    int num_blocks = in_features / 256;  // super-blocks per row

    const nt::BlockQ4_K* row_blocks = reinterpret_cast<const nt::BlockQ4_K*>(W) + row * num_blocks;

    float sum = 0.0f;

    for (int b = tid; b < num_blocks; b += warpSize) {
        const nt::BlockQ4_K& block = row_blocks[b];

        float d = __half2float(*reinterpret_cast<const half*>(&block.d));
        float dmin = __half2float(*reinterpret_cast<const half*>(&block.dmin));

        int base = b * 256;

        // Decode sub-block scales (8 sub-blocks of 32 weights each)
        // Scales are packed in 12 bytes for 8 sub-blocks
        float block_sum = 0.0f;

        for (int sb = 0; sb < 8; sb++) {
            // Decode scale and min for this sub-block
            uint8_t sc, m;
            if (sb < 4) {
                sc = block.scales[sb] & 0x3F;
                m  = block.scales[sb + 4] & 0x3F;
            } else {
                sc = (block.scales[sb + 4] & 0x0F) | ((block.scales[sb - 4] >> 6) << 4);
                m  = (block.scales[sb + 4] >> 4)    | ((block.scales[sb]     >> 6) << 4);
            }

            float sub_d = d * sc;
            float sub_m = dmin * m;
            int sub_base = base + sb * 32;

            float sub_sum = 0.0f;
            float sub_sum_x = 0.0f;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = block.qs[sb * 16 + j];
                int lo = byte & 0x0F;
                int hi = byte >> 4;
                sub_sum += lo * x[sub_base + j] + hi * x[sub_base + j + 16];
                sub_sum_x += x[sub_base + j] + x[sub_base + j + 16];
            }

            block_sum += sub_d * sub_sum - sub_m * sub_sum_x;
        }

        sum += block_sum;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// F16 GEMV (for embeddings / unquantized layers)
// ----------------------------------------------------------
__global__ void gemv_f16_kernel(
    float* __restrict__ y,
    const half* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int tid = threadIdx.x;
    const half* row_w = W + row * in_features;

    float sum = 0.0f;
    for (int i = tid; i < in_features; i += warpSize) {
        sum += __half2float(row_w[i]) * x[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// F32 GEMV
// ----------------------------------------------------------
__global__ void gemv_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= out_features) return;

    int tid = threadIdx.x;
    const float* row_w = W + row * in_features;

    float sum = 0.0f;
    for (int i = tid; i < in_features; i += warpSize) {
        sum += row_w[i] * x[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Batched GEMM for prefill (multiple tokens)
// y[M, N] = x[M, K] * W^T[N, K]   (W stored row-major as [N, K])
// Uses simple tiled approach; optimize in later phases
// ----------------------------------------------------------
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_f32_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,  // [N, K] row-major (transposed)
    int M, int N, int K
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

// ============================================================
// Add bias kernel
// ============================================================
__global__ void add_bias_kernel(
    float* __restrict__ y,
    const float* __restrict__ bias,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        y[i] += bias[i];
    }
}

// ============================================================
// Element-wise multiply (for SwiGLU)
// ============================================================
__global__ void silu_elementwise_mul_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float g = gate[i];
        // SiLU(x) = x * sigmoid(x)
        float silu = g / (1.0f + expf(-g));
        output[i] = silu * up[i];
    }
}

// ============================================================
// Launchers
// ============================================================

void launch_gemv(
    float* y,
    const void* W,
    const float* x,
    int out_features,
    int in_features,
    DType weight_dtype,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    // 4 warps per block, each warp computes one output row
    dim3 block(32, 4);
    dim3 grid((out_features + 3) / 4);

    switch (weight_dtype) {
        case DType::Q4_0:
            gemv_q4_0_kernel<<<grid, block, 0, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::Q8_0:
            gemv_q8_0_kernel<<<grid, block, 0, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::Q4_K_M:
            gemv_q4_k_kernel<<<grid, block, 0, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::F16:
            gemv_f16_kernel<<<grid, block, 0, s>>>(
                y, static_cast<const half*>(W), x, out_features, in_features);
            break;
        case DType::F32:
            gemv_f32_kernel<<<grid, block, 0, s>>>(
                y, static_cast<const float*>(W), x, out_features, in_features);
            break;
        default:
            fprintf(stderr, "Unsupported dtype for GEMV: %s\n", dtype_name(weight_dtype));
            break;
    }
}

void launch_gemm_f32(
    float* C,
    const float* A,
    const float* B,
    int M, int N, int K,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_f32_kernel<TILE, TILE, TILE><<<grid, block, 0, s>>>(C, A, B, M, N, K);
}

void launch_silu_mul(
    float* output,
    const float* gate,
    const float* up,
    int size,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid = (size + block - 1) / block;
    silu_elementwise_mul_kernel<<<grid, block, 0, s>>>(output, gate, up, size);
}

void launch_add_bias(
    float* y,
    const float* bias,
    int size,
    void* stream
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid = (size + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, s>>>(y, bias, size);
}

} // namespace cuda
} // namespace nt
