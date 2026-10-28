#include "ntrpu_q8_block_mac.h"

#include <cstring>
#include <limits>

using namespace nt::ntrpu::block_mac;

extern "C" uint32_t ntrpu_q8_block_mac(
    uint64_t weights_0, uint64_t weights_1, uint64_t weights_2, uint64_t weights_3,
    uint64_t input_0, uint64_t input_1, uint64_t input_2, uint64_t input_3,
    uint16_t weight_scale, double input_scale, double accumulator, double* result) {
#pragma HLS INTERFACE mode=ap_none port=weights_0
#pragma HLS INTERFACE mode=ap_none port=weights_1
#pragma HLS INTERFACE mode=ap_none port=weights_2
#pragma HLS INTERFACE mode=ap_none port=weights_3
#pragma HLS INTERFACE mode=ap_none port=input_0
#pragma HLS INTERFACE mode=ap_none port=input_1
#pragma HLS INTERFACE mode=ap_none port=input_2
#pragma HLS INTERFACE mode=ap_none port=input_3
#pragma HLS INTERFACE mode=ap_none port=weight_scale
#pragma HLS INTERFACE mode=ap_none port=input_scale
#pragma HLS INTERFACE mode=ap_none port=accumulator
#pragma HLS INTERFACE mode=ap_vld port=result
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if ((weight_scale & 0x7c00U) == 0x7c00U ||
        !(input_scale >= 0 && input_scale <= double(std::numeric_limits<float>::max()) / 127.0)) {
        return INVALID_SCALE;
    }
    const double max_double = std::numeric_limits<double>::max();
    if (!(accumulator >= -max_double && accumulator <= max_double)) return INVALID_ACCUMULATOR;

    const uint64_t weights[4] = {weights_0, weights_1, weights_2, weights_3};
    const uint64_t inputs[4] = {input_0, input_1, input_2, input_3};
    int32_t products[32];
#pragma HLS ARRAY_PARTITION variable=weights type=complete
#pragma HLS ARRAY_PARTITION variable=inputs type=complete
#pragma HLS ARRAY_PARTITION variable=products type=complete
    for (uint32_t lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
        const uint32_t shift = (lane % 8) * 8;
        const int w = int((weights[lane / 8] >> shift) & 255U);
        const int x = int((inputs[lane / 8] >> shift) & 255U);
        products[lane] = (w < 128 ? w : w - 256) * (x < 128 ? x : x - 256);
    }
    // All intermediate sums fit INT32, including 32 * (-128) * (-128).
    for (uint32_t width = 16; width != 0; width /= 2) {
#pragma HLS UNROLL
        for (uint32_t lane = 0; lane < width; ++lane) {
#pragma HLS UNROLL
            products[lane] += products[lane + width];
        }
    }
    const uint32_t exponent = (weight_scale >> 10) & 31U;
    const uint32_t fraction = weight_scale & 1023U;
    // Construct an exact IEEE binary64 scale with integer wiring. The earlier
    // reference loop synthesized a long chain of FP64 multiplications by two.
    uint64_t scale_bits = uint64_t(weight_scale & 0x8000U) << 48;
    if (exponent != 0) {
        scale_bits |= uint64_t(exponent + 1008U) << 52;
        scale_bits |= uint64_t(fraction) << 42;
    } else if (fraction != 0) {
        uint32_t leading = 0;
        for (uint32_t bit = 1; bit < 10; ++bit) {
#pragma HLS UNROLL
            if (fraction >= (1U << bit)) leading = bit;
        }
        scale_bits |= uint64_t(leading + 999U) << 52;
        scale_bits |= uint64_t(fraction - (1U << leading)) << (52U - leading);
    }
    double scale;
    static_assert(sizeof(scale) == sizeof(scale_bits) && std::numeric_limits<double>::is_iec559,
                  "Block MAC requires IEEE binary64");
    std::memcpy(&scale, &scale_bits, sizeof(scale));
    // Preserve the existing GEMV operation order; do not sum tile subtotals.
    const double contribution = (scale * input_scale) * double(products[0]);
    const double next = accumulator + contribution;
    if (!(next >= -max_double && next <= max_double)) return NUMERIC_OVERFLOW;
    *result = next;
    return OK;
}
