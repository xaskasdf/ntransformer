#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace rope_table_r1 {
constexpr uint32_t HEAD_DIM = 64;
constexpr uint32_t MAX_HEADS = 40; // 32 Q + 8 K, stored head-major.
constexpr uint32_t MAX_ELEMENTS = HEAD_DIM * MAX_HEADS;
constexpr uint32_t MAX_CONTEXT = 131072;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, INVALID_POSITION = 2,
    INVALID_COEFFICIENT = 3, NONFINITE_INPUT = 4, OUTPUT_OVERFLOW = 5
};
}}}

// GGUF Llama pairs (2*j, 2*j+1). Coefficients are 32 interleaved (cos,sin)
// FP32 pairs. Caller supplies valid, distinct, immutable input/coefficient
// buffers and owns reset/publication. No AXI or external-address validation.
extern "C" uint32_t ntrpu_rope_table_fp32_r1(
    const uint32_t input[2560], const uint32_t coefficients[64],
    uint32_t output[2560], uint32_t heads, uint32_t position,
    uint32_t coefficient_position);
