#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace q8_hls {
constexpr uint32_t MAX_ROWS = 32;
constexpr uint32_t MAX_COLUMNS = 1024;
constexpr uint32_t BLOCK_ELEMENTS = 32;
constexpr uint32_t BLOCK_BYTES = 34;
constexpr uint32_t MAX_BLOCKS = MAX_COLUMNS / BLOCK_ELEMENTS;
constexpr uint32_t MAX_WEIGHT_BYTES = MAX_ROWS * MAX_BLOCKS * BLOCK_BYTES;
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t INVALID_SCALE = 2;
}}}

// Q8_0 row-major bytes: FP16 little-endian scale followed by 32 signed INT8s.
// Activations are already quantized; one nonnegative double scale per 32 values.
// Caller supplies valid, non-overlapping buffers. Columns must be a multiple of 32.
// On error, output is unchanged. Shape errors access no memory.
// Double scaling/accumulation is a correctness baseline, not an area optimization.
extern "C" uint32_t ntrpu_gemv_q8_0(const uint8_t* weights, const int8_t* input,
    const double* input_scales, double* output, uint32_t rows, uint32_t columns);
