#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace fp32_q8_hls {
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t NONFINITE_INPUT = 2;
constexpr uint32_t INVALID_WEIGHT_SCALE = 3;
}}}

// Connected FP32 activation quantization -> Q8_0 GEMV; FP64 output baseline.
// rows=1..32, columns=32..1024 (multiple of 32). Buffers must not overlap.
// Any error leaves the entire output unchanged. Shape errors access no memory.
extern "C" uint32_t ntrpu_gemv_fp32_q8_0(const uint8_t* weights, const float* input,
    double* output, uint32_t rows, uint32_t columns);
