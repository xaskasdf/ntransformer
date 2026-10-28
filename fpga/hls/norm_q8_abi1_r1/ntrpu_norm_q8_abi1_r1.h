#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace norm_q8_r1 {
constexpr uint32_t MAX_ELEMENTS = 2048;
constexpr uint32_t MAX_WORDS = 512;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, INVALID_EPSILON = 2,
    NONFINITE_INPUT = 3, OUTPUT_OVERFLOW = 4, QUANTIZE_ERROR = 5
};
}}}

// RMSNorm -> 32-element Q8 blocks -> 64-byte activation ABI1 records.
// 32..2048 elements, multiple of 32. Output uses low byte first: INT8[32],
// binary64 scale, zero[24]. All buffers distinct, immutable until completion.
// Local memory only; caller owns AXI bounds, epoch, reset and publication.
extern "C" uint32_t ntrpu_norm_q8_abi1_r1(
    const uint32_t input[2048], const uint32_t weight[2048],
    uint64_t output[512], uint32_t elements, uint32_t epsilon_bits);
