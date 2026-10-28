#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace rmsnorm_r1 {
constexpr uint32_t MAX_ELEMENTS = 2048;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, INVALID_EPSILON = 2,
    NONFINITE_INPUT = 3, OUTPUT_OVERFLOW = 4
};
}}}

// Raw IEEE FP32 ports, ordered FP64 sum of squares, checked FP32 output.
// 1..2048 elements; finite positive FP32 epsilon. Independent valid buffers
// are required. No output word is written if validation/calculation fails.
// This local-memory top needs a separate AXI/epoch/reset integration wrapper.
extern "C" uint32_t ntrpu_rmsnorm_fp32_r1(
    const uint32_t input[2048], const uint32_t weight[2048],
    uint32_t output[2048], uint32_t elements, uint32_t epsilon_bits);
