#include "ntrpu_gemv_fp32_q8_0.h"
#include "ntrpu_gemv_q8_0.h"
#include "ntrpu_quantize_q8_0.h"

using namespace nt::ntrpu::fp32_q8_hls;

extern "C" uint32_t ntrpu_gemv_fp32_q8_0(const uint8_t* weights, const float* input,
    double* output, uint32_t rows, uint32_t columns) {
#pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=weights_mem depth=34816
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=input_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=32
#pragma HLS INTERFACE mode=s_axilite port=weights bundle=control
#pragma HLS INTERFACE mode=s_axilite port=input bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=rows bundle=control
#pragma HLS INTERFACE mode=s_axilite port=columns bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    using namespace nt::ntrpu;
    static_assert(q8_hls::MAX_COLUMNS == quantize_hls::MAX_COLUMNS &&
        q8_hls::MAX_BLOCKS == quantize_hls::MAX_BLOCKS, "Incompatible stage limits");
    static_assert(q8_hls::MAX_COLUMNS == 1024 && q8_hls::MAX_ROWS == 32 &&
        q8_hls::MAX_WEIGHT_BYTES == 34816, "Update AXI depths with dimensions");
    if (rows == 0 || rows > q8_hls::MAX_ROWS || columns == 0 ||
        columns > q8_hls::MAX_COLUMNS || columns % q8_hls::BLOCK_ELEMENTS != 0) {
        return INVALID_SHAPE;
    }
    int8_t quantized[q8_hls::MAX_COLUMNS];
    double scales[q8_hls::MAX_BLOCKS];
    // Sequential calls establish producer completion before consumer reads.
    // These arrays are internal hardware storage, not external AXI round trips.
    if (ntrpu_quantize_q8_0(input, quantized, scales, columns) != quantize_hls::OK) {
        return NONFINITE_INPUT;
    }
    const uint32_t status = ntrpu_gemv_q8_0(weights, quantized, scales, output, rows, columns);
    return status == q8_hls::OK ? OK : INVALID_WEIGHT_SCALE;
}
