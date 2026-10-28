#include "ntrpu_gemv_int8.h"

using namespace nt::ntrpu::hls;

extern "C" uint32_t ntrpu_gemv_int8(const int8_t* weights, const int8_t* input,
                                    int32_t* output, uint32_t rows, uint32_t columns) {
#pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=weights_mem depth=32768
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=input_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=32
#pragma HLS INTERFACE mode=s_axilite port=weights bundle=control
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=rows bundle=control
#pragma HLS INTERFACE mode=s_axilite port=columns bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    static_assert(MAX_WEIGHTS == 32768 && MAX_COLUMNS == 1024 && MAX_ROWS == 32,
                  "Update interface depths when changing tile dimensions");
    if (rows == 0 || rows > MAX_ROWS || columns == 0 || columns > MAX_COLUMNS) {
        return INVALID_SHAPE;
    }

    int8_t local_input[MAX_COLUMNS];
    int8_t local_weights[MAX_COLUMNS];
    int32_t accumulators[LANES];
#pragma HLS ARRAY_PARTITION variable=local_input type=cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=local_weights type=cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=accumulators type=complete dim=1

    for (uint32_t column = 0; column < columns; ++column) {
#pragma HLS PIPELINE II=1
        local_input[column] = input[column];
    }
    for (uint32_t row = 0; row < rows; ++row) {
        // Baseline: load a row, then compute it. No overlap or DDR throughput claim.
        for (uint32_t column = 0; column < columns; ++column) {
#pragma HLS PIPELINE II=1
            local_weights[column] = weights[row * columns + column];
        }
        for (uint32_t lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
            accumulators[lane] = 0;
        }
        for (uint32_t base = 0; base < columns; base += LANES) {
#pragma HLS PIPELINE II=1
            for (uint32_t lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                const uint32_t column = base + lane;
                if (column < columns) {
                    accumulators[lane] += static_cast<int32_t>(local_weights[column]) *
                                          static_cast<int32_t>(local_input[column]);
                }
            }
        }
        int32_t sum = 0;
        for (uint32_t lane = 0; lane < LANES; ++lane) {
#pragma HLS PIPELINE II=1
            sum += accumulators[lane];
        }
        // At most 1024 * 16384 = 16777216 in absolute value: INT32 is exact.
        output[row] = sum;
    }
    return OK;
}
