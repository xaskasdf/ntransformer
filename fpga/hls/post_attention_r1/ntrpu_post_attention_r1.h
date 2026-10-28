#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace post_attention_r1 {
constexpr uint32_t MAX_ELEMENTS = 2048;
constexpr uint32_t MAX_WORDS = 512;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, RESIDUAL_NONFINITE = 2,
    RESIDUAL_OVERFLOW = 3, INVALID_EPSILON = 4, NORM_NONFINITE = 5,
    NORM_OVERFLOW = 6, QUANTIZE_ERROR = 7
};
}}}

// Local-memory boundary after O, before FFN. All five arrays must be distinct.
// Preserve inputs until ap_done. On failure neither output is written; on
// success publish both only after ap_done. Reset/epochs/AXI belong to the caller.
// residual_output must survive the FFN so its result can be added to it later.
extern "C" uint32_t ntrpu_post_attention_r1(
    const uint32_t residual_input[2048], const uint32_t branch[2048],
    const uint32_t norm_weight[2048], uint32_t residual_output[2048],
    uint64_t ffn_input[512], uint32_t elements, uint32_t epsilon_bits);
