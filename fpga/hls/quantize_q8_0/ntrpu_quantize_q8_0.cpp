#include "ntrpu_quantize_q8_0.h"

#include <limits>
#include <cstring>

using namespace nt::ntrpu::quantize_hls;

// Widen IEEE binary32 through integer fields: the HLS FP conversion flushes
// subnormal FP32 inputs to zero. Every finite nonzero FP32 is normal in FP64.
static double widen_fp32_bits(uint32_t bits) {
#pragma HLS INLINE
    const uint32_t exponent = (bits >> 23) & 255U;
    uint32_t fraction = bits & 0x007fffffU;
    uint32_t wide_exponent = 0;
    if (exponent == 255U) return 0; // Caller records the error before conversion.
    if (exponent != 0) {
        wide_exponent = exponent + 896U;
    } else if (fraction != 0) {
        wide_exponent = 897U;
        // Normalize the 23-bit subnormal fraction with a five-level shifter.
        for (uint32_t step = 0; step < 5; ++step) {
#pragma HLS UNROLL
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
    double result;
    std::memcpy(&result, &wide_bits, sizeof(result));
    return result;
}

extern "C" uint32_t ntrpu_quantize_q8_0(const float* input, int8_t* output,
    double* output_scales, uint32_t columns) {
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=input_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=output_scales offset=slave bundle=scales_mem depth=32
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output_scales bundle=control
#pragma HLS INTERFACE mode=s_axilite port=columns bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    static_assert(MAX_COLUMNS == 1024 && MAX_BLOCKS == 32, "Update AXI depths with dimensions");
    static_assert(sizeof(float) == 4 && sizeof(double) == 8 &&
        std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559 &&
        std::numeric_limits<float>::digits == 24 && std::numeric_limits<double>::digits == 53,
        "Requires IEEE binary32 and binary64");
    if (columns == 0 || columns > MAX_COLUMNS || columns % BLOCK_ELEMENTS != 0) {
        return INVALID_SHAPE;
    }
    double local[MAX_COLUMNS];
    int8_t quantized[MAX_COLUMNS];
    double scales[MAX_BLOCKS];
    bool nonfinite = false;
    for (uint32_t column = 0; column < columns; ++column) {
#pragma HLS PIPELINE II=1
        const float value = input[column];
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
        local[column] = widen_fp32_bits(bits);
    }
    if (nonfinite) return NONFINITE_INPUT;
    const uint32_t blocks = columns / BLOCK_ELEMENTS;
    for (uint32_t block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE off
        double maximum = 0;
        for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
#pragma HLS PIPELINE off
            const double value = local[block * BLOCK_ELEMENTS + lane];
            const double magnitude = value < 0 ? -value : value;
            if (magnitude > maximum) maximum = magnitude;
        }
        const double scale = maximum / 127.0;
        scales[block] = scale;
        for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
#pragma HLS PIPELINE off
            const uint32_t column = block * BLOCK_ELEMENTS + lane;
            const double value = scale == 0 ? 0 : local[column] / scale;
            const double magnitude = value < 0 ? -value : value;
            // Integer conversion is safe: ratio is bounded by 127, apart from rounding noise.
            int rounded = static_cast<int>(magnitude);
            // Unlike int(magnitude + 0.5), this does not round a value just below a tie up.
            if (magnitude - double(rounded) >= 0.5) ++rounded;
            if (rounded > 127) rounded = 127;
            quantized[column] = static_cast<int8_t>(value < 0 ? -rounded : rounded);
        }
    }
    // No outputs are published until every input has been checked.
    for (uint32_t column = 0; column < columns; ++column) {
#pragma HLS PIPELINE II=1
        output[column] = quantized[column];
    }
    for (uint32_t block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE II=1
        output_scales[block] = scales[block];
    }
    return OK;
}
