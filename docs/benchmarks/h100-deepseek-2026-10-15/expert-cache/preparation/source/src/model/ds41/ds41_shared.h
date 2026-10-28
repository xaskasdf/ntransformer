#pragma once

#include <cuda_runtime_api.h>

namespace nt::ds41::detail {

// Internal asynchronous launches. Buffers and weights remain owned by the
// caller until stream completion. Q8 rows/columns use blocks of 32 elements.
void launch_shared_q8(const void* gate, const void* up, const float* x,
                       float* gate_out, float* up_out, int rows, int cols,
                       bool avx2_order, cudaStream_t stream);
void launch_shared_activation(const float* gate, const float* up, float* h,
                               int n, float limit, cudaStream_t stream);
// Dense router projection using the CPU's two eight-lane accumulators (AVX2)
// or scalar FMA order. BF16 is expanded exactly; all arithmetic stays FP32.
void launch_router(const void* w, const float* x, float* out, int rows, int cols,
                   bool bf16, bool avx2_order, cudaStream_t stream);

} // namespace nt::ds41::detail
