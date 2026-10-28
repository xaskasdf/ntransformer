#include "ntrpu_gemv_q8_0.h"

#include <limits>

using namespace nt::ntrpu::q8_hls;

static double half_scale(uint16_t bits) {
    const uint32_t exponent = (bits >> 10) & 31U;
    const uint32_t fraction = bits & 1023U;
    // Exact powers of two; also preserves FP16 subnormals without FP16 hardware.
    double factor = 1.0 / 16777216.0;
    for (uint32_t step = 1; step < 30; ++step) {
#pragma HLS UNROLL
        if (step < exponent) factor *= 2.0;
    }
    const double magnitude = (exponent == 0 ? fraction : 1024U + fraction) * factor;
    return (bits & 0x8000U) ? -magnitude : magnitude;
}

extern "C" uint32_t ntrpu_gemv_q8_0(const uint8_t* weights, const int8_t* input,
    const double* input_scales, double* output, uint32_t rows, uint32_t columns) {
#pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=weights_mem depth=34816
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=input_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=input_scales offset=slave bundle=scales_mem depth=32
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=32
#pragma HLS INTERFACE mode=s_axilite port=weights bundle=control
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=input_scales bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=rows bundle=control
#pragma HLS INTERFACE mode=s_axilite port=columns bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    static_assert(MAX_WEIGHT_BYTES == 34816 && MAX_COLUMNS == 1024 && MAX_ROWS == 32,
                  "Update AXI depths when changing dimensions");
    if (rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS ||
        columns % BLOCK_ELEMENTS != 0) return INVALID_SHAPE;

    const uint32_t blocks = columns / BLOCK_ELEMENTS;
    int8_t activation[MAX_COLUMNS];
    double scales[MAX_BLOCKS];
    double results[MAX_ROWS];
    int8_t block_weights[BLOCK_ELEMENTS];
#pragma HLS ARRAY_PARTITION variable=activation type=cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=block_weights type=complete dim=1
    for (uint32_t block = 0; block < blocks; ++block) {
        const double scale = input_scales[block];
        // This is the scale range generated from finite FP32 activations.
        // The comparison also rejects NaN and infinity without fast-math assumptions.
        if (!(scale >= 0 && scale <= double(std::numeric_limits<float>::max()) / 127.0)) {
            return INVALID_SCALE;
        }
        scales[block] = scale;
    }
    for (uint32_t column = 0; column < columns; ++column) {
#pragma HLS PIPELINE II=1
        activation[column] = input[column];
    }
    for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint32_t offset = (row * blocks + block) * BLOCK_BYTES;
            const uint16_t bits = uint16_t(weights[offset]) | (uint16_t(weights[offset + 1]) << 8);
            if ((bits & 0x7c00U) == 0x7c00U) return INVALID_SCALE;
            for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
#pragma HLS PIPELINE II=1
                const int value = weights[offset + 2 + lane];
                block_weights[lane] = static_cast<int8_t>(value < 128 ? value : value - 256);
            }
            int32_t dot = 0;
            for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
#pragma HLS UNROLL
                dot += int32_t(block_weights[lane]) * int32_t(activation[block * BLOCK_ELEMENTS + lane]);
            }
            // Scale each block BEFORE accumulation; different blocks have different scales.
            sum += (half_scale(bits) * scales[block]) * double(dot);
        }
        results[row] = sum;
    }
    // Commit only after every scale has passed validation, including the last row.
    for (uint32_t row = 0; row < rows; ++row) {
#pragma HLS PIPELINE II=1
        output[row] = results[row];
    }
    return OK;
}
