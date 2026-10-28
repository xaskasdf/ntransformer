#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace block_mac {
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SCALE = 1;
constexpr uint32_t INVALID_ACCUMULATOR = 2;
constexpr uint32_t NUMERIC_OVERFLOW = 3;
}}}

// One unpacked Q8 block and its matching quantized activations, low byte first.
// Scalar ports use ap_none; hold every input stable from ap_start to ap_done.
// result is valid only on OK. An error leaves result unchanged in C and does
// not assert its ap_vld in RTL. The caller owns row/tag/epoch state and must
// withhold final publication until the complete tile/command is validated.
// Pass the preceding block's result as accumulator, including across K tiles.
extern "C" uint32_t ntrpu_q8_block_mac(
    uint64_t weights_0, uint64_t weights_1, uint64_t weights_2, uint64_t weights_3,
    uint64_t input_0, uint64_t input_1, uint64_t input_2, uint64_t input_3,
    uint16_t weight_scale, double input_scale, double accumulator, double* result);
