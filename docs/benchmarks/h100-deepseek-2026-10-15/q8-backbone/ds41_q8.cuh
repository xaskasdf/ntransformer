// Internal Q8_0 projection kernels. Weight storage and FP32 arithmetic are
// unchanged; the cooperative path only changes how operands reach the warp.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace nt::ds41::detail {
// Q8_0 gemv, warp per row: y[row] = W[row,:] @ x
template<bool Grouped = false>
__global__ void q8_legacy(const uint8_t* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    if constexpr (Grouped) {
        // Independent matrices occupy grid.y; each row keeps the same warp,
        // Q8 block traversal and reduction as the ordinary GEMV.
        W += (size_t)blockIdx.y * rows * nb * 34u;
        x += (size_t)blockIdx.y * cols;
        y += (size_t)blockIdx.y * rows;
    }
    const uint8_t* row = W + (size_t)wid * (size_t)nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh; memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = x + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}


template<bool Grouped = false>
__global__ void q8_cooperative(const uint8_t* __restrict__ W,
                                const float* __restrict__ x,
                                float* __restrict__ y, int rows, int cols) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nb = cols / 32;
    if constexpr (Grouped) {
        W += (size_t)blockIdx.y * rows * nb * 34;
        x += (size_t)blockIdx.y * cols;
        y += (size_t)blockIdx.y * rows;
    }
    extern __shared__ uint32_t storage[];
    uint32_t* words = storage + warp * 272; // 32 packed Q8_0 blocks
    float* inputs = reinterpret_cast<float*>(storage + (blockDim.x / 32) * 272);
    // Even block counts keep every row and grouped matrix word-aligned.
    const uint32_t* row = row_id < rows
        ? reinterpret_cast<const uint32_t*>(W + (size_t)row_id * nb * 34) : nullptr;
    float acc = 0;
    for (int first = 0; first < nb; first += 32) {
        const int blocks = min(32, nb - first);
        if (row) for (int i = lane; i < blocks * 34 / 4; i += 32)
            words[i] = row[first * 34 / 4 + i];
        // Once per CTA instead of reloading x independently for every row.
        // The odd pitch avoids bank conflicts in lane-per-block dot products.
        for (int i = threadIdx.x; i < blocks * 32; i += blockDim.x)
            inputs[(i / 32) * 33 + i % 32] = x[first * 32 + i];
        __syncthreads();
        if (row && lane < blocks) {
            const uint8_t* block = reinterpret_cast<const uint8_t*>(words) + lane * 34;
            uint16_t dh; memcpy(&dh, block, 2);
            const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
            const float* xb = inputs + lane * 33;
            float dot = 0;
#pragma unroll
            for (int i = 0; i < 32; ++i) dot += (float)qs[i] * xb[i];
            acc += d * dot;
        }
        // Inactive rows participate too: a partial CTA must not deadlock.
        __syncthreads();
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (row_id < rows && lane == 0) y[row_id] = acc;
}

inline bool q8_cooperative_supported(int cols) { return cols > 0 && cols % 64 == 0; }

template<bool Grouped = false>
inline void launch_q8_projection(const uint8_t* W, const float* x, float* y,
                                  int rows, int cols, int groups, bool cooperative,
                                  cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 grid((rows + threads / 32 - 1) / (threads / 32), Grouped ? groups : 1);
    if (cooperative && q8_cooperative_supported(cols)) {
        constexpr size_t shared = (threads / 32 * 272 + 32 * 33) * sizeof(uint32_t);
        q8_cooperative<Grouped><<<grid, threads, shared, stream>>>(W, x, y, rows, cols);
    } else q8_legacy<Grouped><<<grid, threads, 0, stream>>>(W, x, y, rows, cols);
}
} // namespace nt::ds41::detail
