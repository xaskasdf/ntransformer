#include "ntrpu_rmsnorm_fp32_r1.h"
#include "ntrpu_swiglu.h"

#if defined(__SYNTHESIS__) && !defined(NTRPU_USE_HLS_MATH)
#error "Use the same AMD math backend in C simulation and synthesis"
#endif
#ifdef NTRPU_USE_HLS_MATH
#include <hls_math.h>
#else
#include <cmath>
#endif

using namespace nt::ntrpu::rmsnorm_r1;

static bool widen_bits(uint32_t bits, double& value) {
#pragma HLS INLINE
    float input;
    std::memcpy(&input, &bits, sizeof(input));
    return nt::ntrpu::swiglu_hls::detail::widen(input, value);
}

extern "C" uint32_t ntrpu_rmsnorm_fp32_r1(
    const uint32_t input[2048], const uint32_t weight[2048],
    uint32_t output[2048], uint32_t elements, uint32_t epsilon_bits) {
#pragma HLS INTERFACE mode=ap_memory port=input storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=weight storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=output storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_none port=elements
#pragma HLS INTERFACE mode=ap_none port=epsilon_bits
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (elements == 0 || elements > MAX_ELEMENTS) return INVALID_SHAPE;
    double epsilon = 0;
    if (!widen_bits(epsilon_bits, epsilon) || !(epsilon > 0)) return INVALID_EPSILON;

    double local[MAX_ELEMENTS];
    uint32_t weights[MAX_ELEMENTS], results[MAX_ELEMENTS];
#pragma HLS BIND_STORAGE variable=local type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=weights type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=results type=ram_1p impl=bram
    bool invalid = false;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        double value = 0, scale = 0;
        const uint32_t raw_weight = weight[i];
        const bool valid_input = widen_bits(input[i], value);
        const bool valid_weight = widen_bits(raw_weight, scale);
        invalid |= !valid_input || !valid_weight;
        local[i] = value;
        weights[i] = raw_weight;
    }
    if (invalid) return NONFINITE_INPUT;

    // Finite FP32 inputs squared and summed over <=2048 values fit FP64.
    // Keep ascending element order; no associative or fast-math reduction.
    double sum = 0;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE off
        const double square = local[i] * local[i];
        sum = sum + square;
    }
    const double mean = sum / double(elements);
    const double variance = mean + epsilon;
#ifdef NTRPU_USE_HLS_MATH
    const double denominator = hls::sqrt(variance);
#else
    const double denominator = std::sqrt(variance);
#endif
    const double inverse = 1.0 / denominator;
    bool overflow = false;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE off
        double scale = 0;
        widen_bits(weights[i], scale);
        const double normalized = local[i] * inverse;
        const double value = normalized * scale;
        float rounded = 0;
        const bool finite = nt::ntrpu::swiglu_hls::detail::narrow(value, rounded);
        overflow |= !finite;
        std::memcpy(&results[i], &rounded, sizeof(rounded));
    }
    if (overflow) return OUTPUT_OVERFLOW;
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = results[i];
    }
    return OK;
}
