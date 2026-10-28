#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace swiglu_block {
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t NONFINITE_INPUT = 2;
constexpr uint32_t NARROW_OVERFLOW = 3;
constexpr uint32_t ACTIVATION_OVERFLOW = 4;
constexpr uint32_t QUANTIZE_ERROR = 5;
}}}

// One block of 1..32 valid gate/up binary64 values from private GEMV output.
// Narrow to FP32, apply the existing SwiGLU contract and quantize 32 lanes,
// padding unused lanes with zero. Output: one activation ABI1 record, low
// byte first: 32 signed INT8, one binary64 scale, then 24 reserved zero bytes.
// All buffers are valid/non-overlapping. Errors leave all eight output words
// untouched. ap_memory ports are local 64-bit memory interfaces, not AXI DMA.
// Hold elements stable until ap_done; publish only after successful return.
extern "C" uint32_t ntrpu_swiglu_q8_block(const uint64_t* gate,
    const uint64_t* up, uint64_t* record, uint32_t elements);
