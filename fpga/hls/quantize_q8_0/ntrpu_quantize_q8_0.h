#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace quantize_hls {
constexpr uint32_t MAX_COLUMNS = 1024;
constexpr uint32_t BLOCK_ELEMENTS = 32;
constexpr uint32_t MAX_BLOCKS = MAX_COLUMNS / BLOCK_ELEMENTS;
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t NONFINITE_INPUT = 2;
}}}

// FP32 activations -> signed INT8 [-127,127] and double scales, per 32 values.
// dX=max(abs(x))/127; nearest integer, ties away from zero; all-zero block -> zeros.
// Shape errors access no memory. Any error leaves both output buffers unchanged.
// Caller guarantees valid, non-overlapping buffers. No fast-math compilation.
extern "C" uint32_t ntrpu_quantize_q8_0(const float* input, int8_t* output,
    double* output_scales, uint32_t columns);
