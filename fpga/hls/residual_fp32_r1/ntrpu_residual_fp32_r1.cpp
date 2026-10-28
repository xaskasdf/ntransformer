#include "ntrpu_residual_fp32_r1.h"

using namespace nt::ntrpu::residual_r1;

static uint32_t shift_jam(uint32_t value, uint32_t shift) {
#pragma HLS INLINE
    if (shift == 0) return value;
    if (shift >= 32) return value != 0;
    const uint32_t mask = (uint32_t(1) << shift) - 1;
    return (value >> shift) | uint32_t((value & mask) != 0);
}

// Integer significands and guard/round/sticky bits; no floating-point IP.
// Inputs are finite. The magnitude ordering also chooses the result sign.
static uint32_t add_finite(uint32_t a, uint32_t b) {
#pragma HLS INLINE
    if ((a & 0x7fffffffU) < (b & 0x7fffffffU)) {
        const uint32_t temp = a; a = b; b = temp;
    }
    const uint32_t sign = a & 0x80000000U;
    uint32_t exponent = (a >> 23) & 255U;
    uint32_t small_exponent = (b >> 23) & 255U;
    uint32_t large = a & 0x7fffffU, small = b & 0x7fffffU;
    if (exponent != 0) large |= 0x800000U;
    else exponent = 1;
    if (small_exponent != 0) small |= 0x800000U;
    else small_exponent = 1;
    large <<= 3;
    small = shift_jam(small << 3, exponent - small_exponent);
    uint32_t sum;
    if (((a ^ b) & 0x80000000U) == 0) {
        sum = large + small;
        if (sum & 0x8000000U) { sum = shift_jam(sum, 1); ++exponent; }
    } else {
        sum = large - small;
        if (sum == 0) return 0; // Exact cancellation is +0 in RNE.
        for (uint32_t i = 0; i < 26; ++i) {
#pragma HLS UNROLL
            if (sum < 0x4000000U && exponent > 1) { sum <<= 1; --exponent; }
        }
    }
    uint32_t rounded = sum >> 3;
    const uint32_t remainder = sum & 7U;
    if (remainder > 4 || (remainder == 4 && (rounded & 1U))) ++rounded;
    if (rounded >= 0x1000000U) { rounded >>= 1; ++exponent; }
    if (exponent >= 255) return sign | 0x7f800000U;
    if (rounded < 0x800000U) exponent = 0;
    return sign | (exponent << 23) | (rounded & 0x7fffffU);
}

extern "C" uint32_t ntrpu_residual_fp32_r1(
    const uint32_t residual[2048], const uint32_t branch[2048],
    uint32_t output[2048], uint32_t elements) {
#pragma HLS INTERFACE mode=ap_memory port=residual storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=branch storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=output storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_none port=elements
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (elements == 0 || elements > MAX_ELEMENTS) return INVALID_SHAPE;
    uint32_t results[MAX_ELEMENTS];
#pragma HLS BIND_STORAGE variable=results type=ram_1p impl=bram
    bool nonfinite = false, overflow = false;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        const uint32_t a = residual[i], b = branch[i];
        const bool finite = (a & 0x7f800000U) != 0x7f800000U &&
                            (b & 0x7f800000U) != 0x7f800000U;
        nonfinite |= !finite;
        const uint32_t result = finite ? add_finite(a, b) : 0;
        overflow |= (result & 0x7f800000U) == 0x7f800000U;
        results[i] = result;
    }
    if (nonfinite) return NONFINITE_INPUT;
    if (overflow) return OUTPUT_OVERFLOW;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = results[i];
    }
    return OK;
}
