#include "ntrpu_ffn_q8_0.h"
#include "ntrpu_gemv_q8_0.h"
#include "ntrpu_quantize_q8_0.h"
#include "ntrpu_swiglu.h"

using namespace nt::ntrpu;
using namespace nt::ntrpu::ffn_hls;

extern "C" uint32_t ntrpu_ffn_q8_0(const uint8_t* gate, const uint8_t* up,
    const uint8_t* down, const float* input, float* output, uint32_t dim, uint32_t hidden) {
#pragma HLS INTERFACE mode=m_axi port=gate offset=slave bundle=weights_mem depth=195840
#pragma HLS INTERFACE mode=m_axi port=up offset=slave bundle=weights_mem depth=195840
#pragma HLS INTERFACE mode=m_axi port=down offset=slave bundle=weights_mem depth=200192
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=input_mem depth=256
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=256
#pragma HLS INTERFACE mode=s_axilite port=gate bundle=control
#pragma HLS INTERFACE mode=s_axilite port=up bundle=control
#pragma HLS INTERFACE mode=s_axilite port=down bundle=control
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=dim bundle=control
#pragma HLS INTERFACE mode=s_axilite port=hidden bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
#pragma HLS ALLOCATION function instances=ntrpu_gemv_q8_0 limit=1
#pragma HLS ALLOCATION function instances=ntrpu_quantize_q8_0 limit=1
    static_assert(MAX_GATE_BYTES == 195840 && MAX_DOWN_BYTES == 200192 &&
        MAX_DIM == 256 && MAX_HIDDEN == 720 && MAX_PADDED == 736, "Update AXI depths");
    if (dim == 0 || dim > MAX_DIM || dim % 32 != 0 || hidden < 2 || hidden > MAX_HIDDEN) {
        return INVALID_SHAPE;
    }
    const uint32_t padded = (hidden + 31) / 32 * 32;
    const uint32_t row_bytes = dim / 32 * 34;
    int8_t quantized[q8_hls::MAX_COLUMNS];
    double scales[q8_hls::MAX_BLOCKS];
    double tile[q8_hls::MAX_ROWS];
    float gate_values[swiglu_hls::MAX_ELEMENTS];
    float up_values[swiglu_hls::MAX_ELEMENTS];
    float activation[quantize_hls::MAX_COLUMNS];
    float results[MAX_DIM];

    if (ntrpu_quantize_q8_0(input, quantized, scales, dim) != quantize_hls::OK) {
        return NONFINITE_INPUT;
    }
    for (uint32_t first = 0; first < hidden; first += q8_hls::MAX_ROWS) {
#pragma HLS PIPELINE off
        const uint32_t rows = hidden - first < q8_hls::MAX_ROWS ? hidden - first : q8_hls::MAX_ROWS;
        if (ntrpu_gemv_q8_0(gate + first * row_bytes, quantized, scales, tile, rows, dim) != q8_hls::OK) {
            return INVALID_WEIGHT_SCALE;
        }
        for (uint32_t r = 0; r < rows; ++r) {
#pragma HLS PIPELINE off
            if (!swiglu_hls::detail::narrow(tile[r], gate_values[first + r])) return NUMERIC_OVERFLOW;
        }
        if (ntrpu_gemv_q8_0(up + first * row_bytes, quantized, scales, tile, rows, dim) != q8_hls::OK) {
            return INVALID_WEIGHT_SCALE;
        }
        for (uint32_t r = 0; r < rows; ++r) {
#pragma HLS PIPELINE off
            if (!swiglu_hls::detail::narrow(tile[r], up_values[first + r])) return NUMERIC_OVERFLOW;
        }
    }
    if (ntrpu_swiglu(gate_values, up_values, activation, hidden) != swiglu_hls::OK) {
        return NUMERIC_OVERFLOW;
    }
    for (uint32_t i = hidden; i < padded; ++i) {
#pragma HLS PIPELINE II=1
        activation[i] = 0.0f;
    }
    if (ntrpu_quantize_q8_0(activation, quantized, scales, padded) != quantize_hls::OK) {
        return NUMERIC_OVERFLOW;
    }
    for (uint32_t first = 0; first < dim; first += q8_hls::MAX_ROWS) {
#pragma HLS PIPELINE off
        if (ntrpu_gemv_q8_0(down + first * (padded / 32) * 34, quantized, scales,
            tile, q8_hls::MAX_ROWS, padded) != q8_hls::OK) return INVALID_WEIGHT_SCALE;
        for (uint32_t r = 0; r < q8_hls::MAX_ROWS; ++r) {
#pragma HLS PIPELINE off
            if (!swiglu_hls::detail::narrow(tile[r], results[first + r])) return NUMERIC_OVERFLOW;
        }
    }
    // All intermediates are private. A late error never publishes partial tiles.
    for (uint32_t i = 0; i < dim; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = results[i];
    }
    return OK;
}
