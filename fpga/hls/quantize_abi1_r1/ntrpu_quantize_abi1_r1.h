#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace quantize_abi1_r1 {
constexpr uint32_t MAX_ELEMENTS = 2048;
constexpr uint32_t MAX_WORDS = 512;
enum Status : uint32_t { OK = 0, INVALID_SHAPE = 1, NONFINITE_INPUT = 2, QUANTIZE_ERROR = 3 };
}}}

// FP32 words -> 64-byte ABI1 records: INT8[32], FP64 scale, zero[24].
// Valid lengths are 32..2048, multiple of 32. Error leaves output unchanged.
// Caller holds distinct input/output memories stable and owns epoch/publication.
extern "C" uint32_t ntrpu_quantize_abi1_r1(
    const uint32_t input[2048], uint64_t output[512], uint32_t elements);
