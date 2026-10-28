#include "ntrpu_rope_table_fp32_r1.h"
#include "ntrpu_swiglu.h"

using namespace nt::ntrpu::rope_table_r1;

static bool widen_bits(uint32_t bits, double& value) {
#pragma HLS INLINE
    float input;
    std::memcpy(&input, &bits, sizeof(input));
    return nt::ntrpu::swiglu_hls::detail::widen(input, value);
}

extern "C" uint32_t ntrpu_rope_table_fp32_r1(
    const uint32_t input[2560], const uint32_t coefficients[64],
    uint32_t output[2560], uint32_t heads, uint32_t position,
    uint32_t coefficient_position) {
#pragma HLS INTERFACE mode=ap_memory port=input storage_type=ram_1p depth=2560
#pragma HLS INTERFACE mode=ap_memory port=coefficients storage_type=ram_1p depth=64
#pragma HLS INTERFACE mode=ap_memory port=output storage_type=ram_1p depth=2560
#pragma HLS INTERFACE mode=ap_none port=heads
#pragma HLS INTERFACE mode=ap_none port=position
#pragma HLS INTERFACE mode=ap_none port=coefficient_position
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (heads == 0 || heads > MAX_HEADS) return INVALID_SHAPE;
    if (position >= MAX_CONTEXT || coefficient_position != position) return INVALID_POSITION;
    double phase[64];
#pragma HLS BIND_STORAGE variable=phase type=ram_1p impl=bram
    bool invalid = false;
    for (uint32_t i = 0; i < HEAD_DIM; i += 2) {
#pragma HLS PIPELINE off
        double c = 0, s = 0;
        const uint32_t raw_c = coefficients[i], raw_s = coefficients[i+1];
        const bool valid_c = widen_bits(raw_c, c), valid_s = widen_bits(raw_s, s);
        const double cc = c*c, ss = s*s;
        const double magnitude = cc+ss;
        invalid |= !valid_c || !valid_s || c < -1 || c > 1 || s < -1 || s > 1;
        invalid |= magnitude < 1.0-0.0000002384185791015625 || magnitude > 1.0+0.0000002384185791015625;
        if (position == 0) invalid |= raw_c != 0x3f800000U || raw_s != 0;
        phase[i] = c; phase[i+1] = s;
    }
    if (invalid) return INVALID_COEFFICIENT;
    const uint32_t elements = heads * HEAD_DIM;
    double local[MAX_ELEMENTS];
    uint32_t results[MAX_ELEMENTS];
#pragma HLS BIND_STORAGE variable=local type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=results type=ram_1p impl=bram
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        double value = 0;
        invalid |= !widen_bits(input[i], value);
        local[i] = value;
    }
    if (invalid) return NONFINITE_INPUT;
    bool overflow = false;
    for (uint32_t h = 0; h < heads; ++h) {
        for (uint32_t i = 0; i < HEAD_DIM; i += 2) {
#pragma HLS PIPELINE off
            const uint32_t index = h*HEAD_DIM+i;
            const double x = local[index], y = local[index+1];
            const double c = phase[i], s = phase[i+1];
            const double xc = x*c, ys = y*s, xs = x*s, yc = y*c;
            const double a = xc-ys, b = xs+yc;
            float rounded_a = 0, rounded_b = 0;
            const bool good_a = nt::ntrpu::swiglu_hls::detail::narrow(a, rounded_a);
            const bool good_b = nt::ntrpu::swiglu_hls::detail::narrow(b, rounded_b);
            overflow |= !good_a || !good_b;
            std::memcpy(&results[index], &rounded_a, sizeof(rounded_a));
            std::memcpy(&results[index+1], &rounded_b, sizeof(rounded_b));
        }
    }
    if (overflow) return OUTPUT_OVERFLOW;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = results[i];
    }
    return OK;
}
