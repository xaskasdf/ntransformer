#include "ds41_shared.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace nt::ds41::detail {
namespace {

template<bool Bf16>
__device__ float router_weight(const void* w, size_t i) {
    if constexpr (Bf16) return __uint_as_float(uint32_t(static_cast<const uint16_t*>(w)[i]) << 16);
    else return static_cast<const float*>(w)[i];
}

template<bool Bf16, bool Avx2>
__global__ void router_projection(const void* w, const float* x, float* out, int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (row >= rows) return;
    const int lane = threadIdx.x & 7;
    const size_t offset = (size_t)row * cols;
    float acc = 0.0f;
    if constexpr (Avx2) {
        float a = 0.0f, b = 0.0f;
        int i = 0;
        for (; i + 16 <= cols; i += 16) {
            a = __fmaf_rn(router_weight<Bf16>(w, offset + i + lane), x[i + lane], a);
            b = __fmaf_rn(router_weight<Bf16>(w, offset + i + lane + 8), x[i + lane + 8], b);
        }
        const unsigned mask = 0xffu << (threadIdx.x & 24);
        acc = __fadd_rn(a, b);
        acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 4, 8));
        acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 1, 8));
        acc = __fadd_rn(acc, __shfl_down_sync(mask, acc, 2, 8));
        if (lane == 0) for (; i < cols; ++i) acc = __fmaf_rn(router_weight<Bf16>(w, offset + i), x[i], acc);
    } else if (lane == 0) {
        for (int i = 0; i < cols; ++i) acc = __fmaf_rn(router_weight<Bf16>(w, offset + i), x[i], acc);
    }
    if (lane == 0) out[row] = acc;
}

// Compute blocks independently, then accumulate their scaled results in the
// CPU's original block order. The eight-lane reduction mirrors dot_q8_0's AVX2
// multiply, three FMAs, low/high add and two horizontal adds exactly.
template<bool Avx2>
__global__ void shared_q8(const uint8_t* gate, const uint8_t* up, const float* x,
                          float* gate_out, float* up_out, int cols) {
    extern __shared__ float scratch[];
    const int nb = cols / 32;
    float* sums = scratch;
    float* scales = scratch + nb;
    const uint8_t* row = (blockIdx.y ? up : gate) + (size_t)blockIdx.x * nb * 34;
    const int lane = threadIdx.x & 7;
    for (int b = threadIdx.x / 8; b < nb; b += blockDim.x / 8) {
        const uint8_t* block = row + b * 34;
        const int8_t* q = reinterpret_cast<const int8_t*>(block + 2);
        const float* xb = x + b * 32;
        float sum;
        if constexpr (Avx2) {
            // Membership is fixed by the eight-lane Q8 block, independent of
            // which lanes happen to be scheduled together at this instruction.
            const unsigned mask = 0xffu << (threadIdx.x & 24);
            sum = __fmul_rn(float(q[lane]), xb[lane]);
            sum = __fmaf_rn(float(q[lane + 8]), xb[lane + 8], sum);
            sum = __fmaf_rn(float(q[lane + 16]), xb[lane + 16], sum);
            sum = __fmaf_rn(float(q[lane + 24]), xb[lane + 24], sum);
            sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 4, 8));
            sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 1, 8));
            sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 2, 8));
        } else if (lane == 0) {
            sum = 0.0f;
#pragma unroll 1
            for (int i = 0; i < 32; ++i) sum = __fmaf_rn(float(q[i]), xb[i], sum);
        }
        if (lane == 0) {
            sums[b] = sum;
            scales[b] = __half2float(*reinterpret_cast<const __half*>(block));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
#pragma unroll 1
        for (int b = 0; b < nb; ++b) acc = __fmaf_rn(scales[b], sums[b], acc);
        (blockIdx.y ? up_out : gate_out)[blockIdx.x] = acc;
    }
}

__global__ void shared_activation(const float* gate, const float* up, float* h,
                                  int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (limit > 1e-6f) {
        // Comparisons match std::min/max, including equal signed zeros.
        g = limit < g ? limit : g;
        u = limit < u ? limit : u;
        u = -limit < u ? u : -limit;
    }
    // Round a double-precision exponential to FP32 before the original FP32
    // add/divide/multiply. This reduces library approximation differences
    // versus host expf without changing tensor storage or the Q8 reductions.
    const float exponential = (float)exp(-(double)g);
    h[i] = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, exponential)), u);
}
} // namespace

void launch_router(const void* w, const float* x, float* out, int rows, int cols,
                   bool bf16, bool avx2_order, cudaStream_t stream) {
    // HC matrices have only 24 long rows. Give each its own block instead of
    // concentrating all that work in six blocks. Larger routers pack four rows.
    const int threads = rows <= 32 ? 8 : 32;
    const int blocks = (rows + threads / 8 - 1) / (threads / 8);
    if (bf16) {
        if (avx2_order) router_projection<true, true><<<blocks, threads, 0, stream>>>(w, x, out, rows, cols);
        else router_projection<true, false><<<blocks, threads, 0, stream>>>(w, x, out, rows, cols);
    } else {
        if (avx2_order) router_projection<false, true><<<blocks, threads, 0, stream>>>(w, x, out, rows, cols);
        else router_projection<false, false><<<blocks, threads, 0, stream>>>(w, x, out, rows, cols);
    }
}

void launch_shared_q8(const void* gate, const void* up, const float* x,
                       float* gate_out, float* up_out, int rows, int cols,
                       bool avx2_order, cudaStream_t stream) {
    const dim3 grid(rows, up ? 2 : 1);
    const size_t smem = 2 * (cols / 32) * sizeof(float);
    if (avx2_order) shared_q8<true><<<grid, 256, smem, stream>>>(
        static_cast<const uint8_t*>(gate), static_cast<const uint8_t*>(up), x, gate_out, up_out, cols);
    else shared_q8<false><<<grid, 256, smem, stream>>>(
        static_cast<const uint8_t*>(gate), static_cast<const uint8_t*>(up), x, gate_out, up_out, cols);
}

void launch_shared_activation(const float* gate, const float* up, float* h,
                               int n, float limit, cudaStream_t stream) {
    shared_activation<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, h, n, limit);
}
} // namespace nt::ds41::detail
