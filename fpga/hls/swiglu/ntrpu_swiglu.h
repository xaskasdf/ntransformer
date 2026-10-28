#pragma once

#include <cstdint>
#include <cstring>
#include <limits>

namespace nt { namespace ntrpu { namespace swiglu_hls {
constexpr uint32_t MAX_ELEMENTS = 1024;
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t NONFINITE_INPUT = 2;
constexpr uint32_t OUTPUT_OVERFLOW = 3;

// Integer IEEE conversions avoid the HLS conversion operators' subnormal flush.
// These helpers require IEEE binary32/binary64 and round-to-nearest, ties-to-even.
namespace detail {
static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
    std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
    "IEEE binary32/binary64 required");

inline bool widen(float value, double& result) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 255U;
    uint32_t fraction = bits & 0x007fffffU;
    if (exponent == 255U) return false;
    uint32_t wide_exponent = 0;
    if (exponent != 0) {
        wide_exponent = exponent + 896U;
    } else if (fraction != 0) {
        wide_exponent = 897U;
        for (uint32_t step = 0; step < 5; ++step) {
#ifdef __SYNTHESIS__
#pragma HLS UNROLL
#endif
            const uint32_t shift = 16U >> step;
            if (fraction < (1U << (24U - shift))) {
                fraction <<= shift;
                wide_exponent -= shift;
            }
        }
        fraction &= 0x007fffffU;
    }
    const uint64_t wide_bits = (uint64_t(bits & 0x80000000U) << 32) |
        (uint64_t(wide_exponent) << 52) | (uint64_t(fraction) << 29);
    std::memcpy(&result, &wide_bits, sizeof(result));
    return true;
}

// Reject values outside the finite FP32 range, matching the CPU FFN contract.
// Successful conversion includes signed zero and gradual underflow.
inline bool narrow(double value, float& result) {
#ifdef __SYNTHESIS__
#pragma HLS INLINE
#endif
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint64_t magnitude = bits & 0x7fffffffffffffffULL;
    if (magnitude > 0x47efffffe0000000ULL) return false; // double(FLT_MAX)
    const int power = int((bits >> 52) & 2047U) - 1023;
    uint32_t out = uint32_t(bits >> 32) & 0x80000000U;
    if (power >= -150) {
        const uint64_t significand = (bits & 0x000fffffffffffffULL) | (1ULL << 52);
        const uint32_t shift = power >= -126 ? 29U : uint32_t(-97 - power);
        uint32_t rounded = uint32_t(significand >> shift);
        const uint64_t remainder = significand & ((1ULL << shift) - 1);
        const uint64_t halfway = 1ULL << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U))) ++rounded;
        out |= power >= -126 ? (uint32_t(power + 127) << 23) + rounded - (1U << 23) : rounded;
    }
    std::memcpy(&result, &out, sizeof(result));
    return true;
}
} // namespace detail
}}}

// Stable SiLU(gate) * up with FP64 intermediates, then checked FP32 rounding.
// Valid, non-overlapping buffers required. Errors leave the entire output intact.
// HLS builds define NTRPU_USE_HLS_MATH for both C simulation and synthesis.
extern "C" uint32_t ntrpu_swiglu(const float* gate, const float* up, float* output,
    uint32_t elements);
