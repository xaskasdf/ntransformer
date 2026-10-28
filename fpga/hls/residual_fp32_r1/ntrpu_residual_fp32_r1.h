#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace residual_r1 {
constexpr uint32_t MAX_ELEMENTS = 2048;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, NONFINITE_INPUT = 2, OUTPUT_OVERFLOW = 3
};
}}}

// Finite binary32 addition, round-to-nearest ties-to-even, gradual underflow.
// Distinct input/output arrays, immutable inputs until completion. Local RAM
// only; the caller owns AXI bounds, reset, epochs and external publication.
extern "C" uint32_t ntrpu_residual_fp32_r1(
    const uint32_t residual[2048], const uint32_t branch[2048],
    uint32_t output[2048], uint32_t elements);
