#include "ntrpu_swiglu_q8_block.h"
#include "ntrpu_swiglu.h"
#include "ntrpu_quantize_q8_0.h"

#include <cstring>

using namespace nt::ntrpu;

extern "C" uint32_t ntrpu_swiglu_q8_block(const uint64_t* gate,
    const uint64_t* up, uint64_t* record, uint32_t elements) {
#pragma HLS INTERFACE mode=ap_memory port=gate depth=32
#pragma HLS INTERFACE mode=ap_memory port=up depth=32
#pragma HLS INTERFACE mode=ap_memory port=record depth=8
#pragma HLS INTERFACE mode=ap_none port=elements
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (elements == 0 || elements > 32) return swiglu_block::INVALID_SHAPE;
    float g[32], u[32], activation[32];
    int8_t quantized[32];
    double scale[1];
    uint64_t packed[8] = {};
    bool nonfinite = false, overflow = false;
    for (uint32_t i = 0; i < 32; ++i) {
#pragma HLS PIPELINE off
        g[i] = 0; u[i] = 0;
        if (i < elements) {
            const uint64_t gb = gate[i], ub = up[i];
            const bool bad = ((gb >> 52) & 2047U) == 2047U || ((ub >> 52) & 2047U) == 2047U;
            nonfinite |= bad;
            if (!bad) {
                double gd, ud;
                std::memcpy(&gd, &gb, sizeof(gd));
                std::memcpy(&ud, &ub, sizeof(ud));
                const bool vg = swiglu_hls::detail::narrow(gd, g[i]);
                const bool vu = swiglu_hls::detail::narrow(ud, u[i]);
                overflow |= !vg || !vu;
            }
        }
    }
    // Validate the whole block before any arithmetic or externally visible write.
    if (nonfinite) return swiglu_block::NONFINITE_INPUT;
    if (overflow) return swiglu_block::NARROW_OVERFLOW;
    if (ntrpu_swiglu(g, u, activation, 32) != swiglu_hls::OK)
        return swiglu_block::ACTIVATION_OVERFLOW;
    if (ntrpu_quantize_q8_0(activation, quantized, scale, 32) != quantize_hls::OK)
        return swiglu_block::QUANTIZE_ERROR;
    for (uint32_t i = 0; i < 32; ++i) {
#pragma HLS PIPELINE off
        packed[i / 8] |= uint64_t(uint8_t(quantized[i])) << ((i % 8) * 8);
    }
    std::memcpy(&packed[4], &scale[0], sizeof(double));
    for (uint32_t i = 0; i < 8; ++i) {
#pragma HLS PIPELINE II=1
        record[i] = packed[i];
    }
    return swiglu_block::OK;
}
