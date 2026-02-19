#include "../core/types.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace nt {
namespace cuda {

// ============================================================
// Optimized GEMV kernels for LLM decode inference
//
// Key optimizations over baseline:
// 1. Input vector x cached in shared memory - eliminates redundant
//    global memory reads across warps (8x bandwidth savings)
// 2. 8 warps per block for better latency hiding and SM occupancy
// 3. Dynamic shared memory for variable input dimensions
// 4. Vectorized half2 loads for F16 weights (LM head)
//
// For decode: y[out] = W[out, in] * x[in]
// Each warp computes one output row via dot product + warp reduction.
// All warps in a block share the same x vector via shared memory.
// ============================================================

static constexpr int GEMV_WARPS = 8;

// One-time extended shared memory configuration
static bool s_smem_configured = false;

// ----------------------------------------------------------
// Q4_0 GEMV: 32 weights/block, FP16 scale + 16 nibble bytes
// ----------------------------------------------------------
__global__ void gemv_q4_0_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int flat_id = warp_id * 32 + tid;
    const int nthreads = blockDim.y * 32;

    // Cooperatively load x into shared memory
    for (int i = flat_id; i < in_features; i += nthreads) {
        sx[i] = x[i];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= out_features) return;

    const int num_blocks = in_features / 32;
    const nt::BlockQ4_0* row_blocks = reinterpret_cast<const nt::BlockQ4_0*>(W) + row * num_blocks;

    float sum = 0.0f;

    for (int b = tid; b < num_blocks; b += 32) {
        const nt::BlockQ4_0& block = row_blocks[b];
        float d = __half2float(*reinterpret_cast<const half*>(&block.d));
        const int base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t byte = block.qs[j];
            int8_t lo = (byte & 0x0F) - 8;
            int8_t hi = (byte >> 4) - 8;
            block_sum += lo * sx[base + j] + hi * sx[base + j + 16];
        }

        sum += d * block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Q8_0 GEMV: 32 weights/block, FP16 scale + 32 int8
// Primary kernel for Q8_0 quantized models
// ----------------------------------------------------------
__global__ void gemv_q8_0_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int flat_id = warp_id * 32 + tid;
    const int nthreads = blockDim.y * 32;

    // Cooperatively load x into shared memory (vectorized float4)
    {
        const int n_float4 = in_features / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* sx4 = reinterpret_cast<float4*>(sx);
        for (int i = flat_id; i < n_float4; i += nthreads) {
            sx4[i] = x4[i];
        }
        // Handle remainder if in_features not divisible by 4
        for (int i = n_float4 * 4 + flat_id; i < in_features; i += nthreads) {
            sx[i] = x[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= out_features) return;

    const int num_blocks = in_features / 32;
    const nt::BlockQ8_0* row_blocks = reinterpret_cast<const nt::BlockQ8_0*>(W) + row * num_blocks;

    float sum = 0.0f;

    for (int b = tid; b < num_blocks; b += 32) {
        const nt::BlockQ8_0& block = row_blocks[b];
        float d = __half2float(*reinterpret_cast<const half*>(&block.d));
        const int base = b * 32;

        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            block_sum += block.qs[j] * sx[base + j];
        }

        sum += d * block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Q4_K_M GEMV: 256 weights per super-block
// ----------------------------------------------------------
__global__ void gemv_q4_k_kernel(
    float* __restrict__ y,
    const void* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int flat_id = warp_id * 32 + tid;
    const int nthreads = blockDim.y * 32;

    // Cooperatively load x into shared memory
    for (int i = flat_id; i < in_features; i += nthreads) {
        sx[i] = x[i];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= out_features) return;

    const int num_blocks = in_features / 256;
    const nt::BlockQ4_K* row_blocks = reinterpret_cast<const nt::BlockQ4_K*>(W) + row * num_blocks;

    float sum = 0.0f;

    for (int b = tid; b < num_blocks; b += 32) {
        const nt::BlockQ4_K& block = row_blocks[b];

        float d = __half2float(*reinterpret_cast<const half*>(&block.d));
        float dmin = __half2float(*reinterpret_cast<const half*>(&block.dmin));

        const int base = b * 256;

        float block_sum = 0.0f;

        for (int sb = 0; sb < 8; sb++) {
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
                sub_sum += lo * sx[sub_base + j] + hi * sx[sub_base + j + 16];
                sub_sum_x += sx[sub_base + j] + sx[sub_base + j + 16];
            }

            block_sum += sub_d * sub_sum - sub_m * sub_sum_x;
        }

        sum += block_sum;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// F16 GEMV: vectorized half2 weight loads
// Used for LM head (output.weight in F16)
// ----------------------------------------------------------
__global__ void gemv_f16_kernel(
    float* __restrict__ y,
    const half* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int flat_id = warp_id * 32 + tid;
    const int nthreads = blockDim.y * 32;

    // Cooperatively load x into shared memory (vectorized float4)
    {
        const int n_float4 = in_features / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* sx4 = reinterpret_cast<float4*>(sx);
        for (int i = flat_id; i < n_float4; i += nthreads) {
            sx4[i] = x4[i];
        }
        for (int i = n_float4 * 4 + flat_id; i < in_features; i += nthreads) {
            sx[i] = x[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= out_features) return;

    const half* row_w = W + (long long)row * in_features;

    float sum = 0.0f;

    // Process 8 elements per iteration using half2 vectorized loads
    const int in_features_8 = (in_features / 8) * 8;
    for (int i = tid * 8; i < in_features_8; i += 32 * 8) {
        const half2* w2 = reinterpret_cast<const half2*>(row_w + i);
        half2 h0 = w2[0];
        half2 h1 = w2[1];
        half2 h2 = w2[2];
        half2 h3 = w2[3];
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);
        sum += f0.x * sx[i]   + f0.y * sx[i+1] +
               f1.x * sx[i+2] + f1.y * sx[i+3] +
               f2.x * sx[i+4] + f2.y * sx[i+5] +
               f3.x * sx[i+6] + f3.y * sx[i+7];
    }
    // Handle remainder
    for (int i = in_features_8 + tid; i < in_features; i += 32) {
        sum += __half2float(row_w[i]) * sx[i];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// F32 GEMV: vectorized float4 loads
// ----------------------------------------------------------
__global__ void gemv_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ W,
    const float* __restrict__ x,
    int out_features,
    int in_features
) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int flat_id = warp_id * 32 + tid;
    const int nthreads = blockDim.y * 32;

    // Cooperatively load x into shared memory
    {
        const int n_float4 = in_features / 4;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* sx4 = reinterpret_cast<float4*>(sx);
        for (int i = flat_id; i < n_float4; i += nthreads) {
            sx4[i] = x4[i];
        }
        for (int i = n_float4 * 4 + flat_id; i < in_features; i += nthreads) {
            sx[i] = x[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.y + warp_id;
    if (row >= out_features) return;

    const float* row_w = W + (long long)row * in_features;

    float sum = 0.0f;

    // Vectorized float4 loads for both weights and shared-memory x
    const int in_features_4 = (in_features / 4) * 4;
    for (int i = tid * 4; i < in_features_4; i += 32 * 4) {
        float4 wv = *reinterpret_cast<const float4*>(row_w + i);
        sum += wv.x * sx[i] + wv.y * sx[i+1] + wv.z * sx[i+2] + wv.w * sx[i+3];
    }
    for (int i = in_features_4 + tid; i < in_features; i += 32) {
        sum += row_w[i] * sx[i];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

// ----------------------------------------------------------
// Batched GEMM for prefill (multiple tokens)
// y[M, N] = x[M, K] * W^T[N, K]   (W stored row-major as [N, K])
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
// SwiGLU: output = SiLU(gate) * up
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
        float silu = g / (1.0f + expf(-g));
        output[i] = silu * up[i];
    }
}

// ============================================================
// Launchers
// ============================================================

static void ensure_smem_config() {
    if (s_smem_configured) return;
    // Allow up to 64KB dynamic shared memory for FFN layers
    // (in_features=14336 needs 14336*4 = 56KB)
    constexpr int MAX_SMEM = 64 * 1024;
    cudaFuncSetAttribute((const void*)gemv_q4_0_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM);
    cudaFuncSetAttribute((const void*)gemv_q8_0_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM);
    cudaFuncSetAttribute((const void*)gemv_q4_k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM);
    cudaFuncSetAttribute((const void*)gemv_f16_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM);
    cudaFuncSetAttribute((const void*)gemv_f32_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SMEM);
    s_smem_configured = true;
}

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
    ensure_smem_config();

    dim3 block(32, GEMV_WARPS);
    dim3 grid((out_features + GEMV_WARPS - 1) / GEMV_WARPS);
    size_t smem = (size_t)in_features * sizeof(float);

    switch (weight_dtype) {
        case DType::Q4_0:
            gemv_q4_0_kernel<<<grid, block, smem, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::Q8_0:
            gemv_q8_0_kernel<<<grid, block, smem, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::Q4_K_M:
            gemv_q4_k_kernel<<<grid, block, smem, s>>>(y, W, x, out_features, in_features);
            break;
        case DType::F16:
            gemv_f16_kernel<<<grid, block, smem, s>>>(
                y, static_cast<const half*>(W), x, out_features, in_features);
            break;
        case DType::F32:
            gemv_f32_kernel<<<grid, block, smem, s>>>(
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
