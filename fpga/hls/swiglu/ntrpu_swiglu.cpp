#include "ntrpu_swiglu.h"

#if defined(__SYNTHESIS__) && !defined(NTRPU_USE_HLS_MATH)
#error "Define NTRPU_USE_HLS_MATH for C simulation and synthesis alike"
#endif

#ifdef NTRPU_USE_HLS_MATH
#include <hls_math.h>
#else
#include <cmath>
#endif

using namespace nt::ntrpu::swiglu_hls;

extern "C" uint32_t ntrpu_swiglu(const float* gate, const float* up, float* output,
    uint32_t elements) {
#pragma HLS INTERFACE mode=m_axi port=gate offset=slave bundle=gate_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=up offset=slave bundle=up_mem depth=1024
#pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=output_mem depth=1024
#pragma HLS INTERFACE mode=s_axilite port=gate bundle=control
#pragma HLS INTERFACE mode=s_axilite port=up bundle=control
#pragma HLS INTERFACE mode=s_axilite port=output bundle=control
#pragma HLS INTERFACE mode=s_axilite port=elements bundle=control
#pragma HLS INTERFACE mode=s_axilite port=return bundle=control
    if (elements == 0 || elements > MAX_ELEMENTS) return INVALID_SHAPE;
    float results[MAX_ELEMENTS];
    uint32_t status = OK;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE off
        double g = 0, u = 0;
        const bool valid_gate = nt::ntrpu::swiglu_hls::detail::widen(gate[i], g);
        const bool valid_up = nt::ntrpu::swiglu_hls::detail::widen(up[i], u);
        if (!valid_gate || !valid_up) {
            status = NONFINITE_INPUT;
        } else {
            const double magnitude = g < 0 ? -g : g;
            // Beyond 256, the negative branch rounds to FP32 zero even for
            // FLT_MAX up. The positive branch's sigmoid rounds to one.
            double e = 0;
            if (magnitude <= 256.0) {
#ifdef NTRPU_USE_HLS_MATH
                e = hls::exp(-magnitude);
#else
                e = std::exp(-magnitude);
#endif
            }
            const double sigmoid = g >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
            if (!nt::ntrpu::swiglu_hls::detail::narrow((g * sigmoid) * u, results[i]) && status == OK) {
                status = OUTPUT_OVERFLOW;
            }
        }
    }
    if (status != OK) return status;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = results[i];
    }
    return OK;
}
