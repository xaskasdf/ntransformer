#include "ntrpu_attention_tile_fp32_r1.h"
#include "ntrpu_swiglu.h"

#if defined(__SYNTHESIS__) && !defined(NTRPU_USE_HLS_MATH)
#error "Use the same AMD math backend in C simulation and synthesis"
#endif
#ifdef NTRPU_USE_HLS_MATH
#include <hls_math.h>
#else
#include <cmath>
#endif

using namespace nt::ntrpu::attention_tile_r1;

static bool finite32(uint32_t word) {
#pragma HLS INLINE
    return (word & 0x7f800000U) != 0x7f800000U;
}
static bool finite64(uint64_t word) {
#pragma HLS INLINE
    return (word & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
static double value64(uint64_t word) {
#pragma HLS INLINE
    double value; std::memcpy(&value, &word, 8); return value;
}
static uint64_t bits64(double value) {
#pragma HLS INLINE
    uint64_t word; std::memcpy(&word, &value, 8); return word;
}
static double value32(uint32_t word) {
#pragma HLS INLINE
    float value; double result = 0; std::memcpy(&value, &word, 4);
    nt::ntrpu::swiglu_hls::detail::widen(value, result); return result;
}
static double exponential(double value) {
#pragma HLS INLINE off
    if (value == 0) return 1;
#ifdef NTRPU_USE_HLS_MATH
    return hls::exp(value);
#else
    return std::exp(value);
#endif
}

extern "C" uint32_t ntrpu_attention_tile_fp32_r1(
    const uint32_t query[64], const uint32_t keys[4096], const uint32_t values[4096],
    const uint64_t state_input[104], uint64_t state_output[104], uint32_t output[64],
    uint32_t tile_start, uint32_t tile_count, uint32_t query_position,
    uint32_t layer, uint32_t query_head, uint32_t kv_head,
    uint64_t sequence, uint64_t epoch) {
#pragma HLS INTERFACE mode=ap_memory port=query storage_type=ram_1p depth=64
#pragma HLS INTERFACE mode=ap_memory port=keys storage_type=ram_1p depth=4096
#pragma HLS INTERFACE mode=ap_memory port=values storage_type=ram_1p depth=4096
#pragma HLS INTERFACE mode=ap_memory port=state_input storage_type=ram_1p depth=104
#pragma HLS INTERFACE mode=ap_memory port=state_output storage_type=ram_1p depth=104
#pragma HLS INTERFACE mode=ap_memory port=output storage_type=ram_1p depth=64
#pragma HLS INTERFACE mode=ap_none port=tile_start
#pragma HLS INTERFACE mode=ap_none port=tile_count
#pragma HLS INTERFACE mode=ap_none port=query_position
#pragma HLS INTERFACE mode=ap_none port=layer
#pragma HLS INTERFACE mode=ap_none port=query_head
#pragma HLS INTERFACE mode=ap_none port=kv_head
#pragma HLS INTERFACE mode=ap_none port=sequence
#pragma HLS INTERFACE mode=ap_none port=epoch
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    const uint64_t end = uint64_t(tile_start) + tile_count;
    if (tile_count == 0 || tile_count > TILE_TOKENS || query_position >= MAX_CONTEXT ||
        end > uint64_t(query_position) + 1) return INVALID_SHAPE;
    if (layer >= 16 || query_head >= 32 || kv_head != query_head / 4 ||
        sequence == 0 || epoch == 0) return INVALID_IDENTITY;
    const uint64_t identity = (uint64_t(layer) << 32) | query_head;
    const double maximum_fp32 = value32(0x7f7fffffU);
    double maximum = 0, normalizer = 0, accumulator[64] = {};
    uint64_t next_state[104];
    uint32_t result[64], query_bits[64], local_keys[4096], local_values[4096];
    double q[64];
#pragma HLS BIND_STORAGE variable=accumulator type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=next_state type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=local_keys type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=local_values type=ram_1p impl=bram
    bool nonfinite = false;
    for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE II=1
        const uint32_t word = query[d]; query_bits[d] = word;
        nonfinite |= !finite32(word); q[d] = value32(word);
    }
    if (tile_start != 0) {
        if (state_input[0] != STATE_MAGIC || state_input[1] != sequence ||
            state_input[2] != epoch || state_input[3] != query_position ||
            state_input[4] != identity || state_input[5] != tile_start) return INVALID_STATE;
        bool bad_state = !finite64(state_input[6]) || !finite64(state_input[7]);
        maximum = value64(state_input[6]); normalizer = value64(state_input[7]);
        const double score_bound = (maximum_fp32 * maximum_fp32) * 8.0;
        bad_state |= maximum < -score_bound || maximum > score_bound ||
                     normalizer < 1.0 || normalizer > double(tile_start);
        for (uint32_t i = 0; i < 32; ++i) {
#pragma HLS PIPELINE II=1
            const uint64_t expected = uint64_t(query_bits[2*i]) | (uint64_t(query_bits[2*i+1]) << 32);
            bad_state |= state_input[8+i] != expected;
        }
        const double bound = normalizer * maximum_fp32;
        for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE II=1
            const uint64_t word = state_input[40+d];
            accumulator[d] = value64(word);
            // A small FP64 allowance accounts for accumulated rounding.
            const double tolerance = bound * 0.00000000023283064365386962890625;
            bad_state |= !finite64(word) || accumulator[d] < -bound-tolerance || accumulator[d] > bound+tolerance;
        }
        if (bad_state) return INVALID_STATE;
    }
    for (uint32_t i = 0; i < tile_count * 64; ++i) {
#pragma HLS PIPELINE II=1
        const uint32_t k = keys[i], v = values[i];
        nonfinite |= !finite32(k) || !finite32(v);
        local_keys[i] = k; local_values[i] = v;
    }
    if (nonfinite) return NONFINITE_INPUT;
    for (uint32_t token = 0; token < tile_count; ++token) {
#pragma HLS PIPELINE off
        double dot = 0;
        for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE off
            const double product = q[d] * value32(local_keys[token*64+d]);
            dot = dot + product;
        }
        const double score = dot * 0.125;
        if (tile_start == 0 && token == 0) {
            maximum = score; normalizer = 1;
            for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE II=1
                accumulator[d] = value32(local_values[d]);
            }
        } else {
            const double new_maximum = score > maximum ? score : maximum;
            const double old_factor = exponential(maximum - new_maximum);
            const double new_factor = exponential(score - new_maximum);
            if (!finite64(bits64(old_factor)) || !finite64(bits64(new_factor)) ||
                old_factor < 0 || old_factor > 1 || new_factor < 0 || new_factor > 1) return NUMERICAL_ERROR;
            const double previous_mass = normalizer * old_factor;
            normalizer = previous_mass + new_factor; maximum = new_maximum;
            for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE off
                const double previous = accumulator[d] * old_factor;
                const double current = value32(local_values[token*64+d]) * new_factor;
                accumulator[d] = previous + current;
            }
        }
    }
    if (!finite64(bits64(maximum)) || !finite64(bits64(normalizer)) ||
        normalizer < 1 || normalizer > double(end)) return NUMERICAL_ERROR;
    const bool final_tile = end == uint64_t(query_position) + 1;
    for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE off
        if (!finite64(bits64(accumulator[d]))) return NUMERICAL_ERROR;
        if (final_tile) {
            double normalized = accumulator[d] / normalizer;
            // A finite convex mean cannot exceed the input FP32 domain. The
            // two separately rounded mass sums can place it just outside that
            // domain. Accept at most the same 2^-32 relative guard as state
            // validation, far below half an FP32 ULP here, and round to FLT_MAX.
            // This is a domain guard, not a general attention accuracy bound.
            const double excess = maximum_fp32 * 0.00000000023283064365386962890625;
            if (normalized > maximum_fp32) {
                if (normalized > maximum_fp32 + excess) return NUMERICAL_ERROR;
                normalized = maximum_fp32;
            } else if (normalized < -maximum_fp32) {
                if (normalized < -maximum_fp32 - excess) return NUMERICAL_ERROR;
                normalized = -maximum_fp32;
            }
            float rounded = 0;
            if (!nt::ntrpu::swiglu_hls::detail::narrow(normalized, rounded)) return NUMERICAL_ERROR;
            std::memcpy(&result[d], &rounded, 4);
        }
    }
    next_state[0] = STATE_MAGIC; next_state[1] = sequence; next_state[2] = epoch;
    next_state[3] = query_position; next_state[4] = identity; next_state[5] = end;
    next_state[6] = bits64(maximum); next_state[7] = bits64(normalizer);
    for (uint32_t i = 0; i < 32; ++i) {
#pragma HLS PIPELINE II=1
        next_state[8+i] = uint64_t(query_bits[2*i]) | (uint64_t(query_bits[2*i+1]) << 32);
    }
    for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE II=1
        next_state[40+d] = bits64(accumulator[d]);
    }
    for (uint32_t i = 0; i < 104; ++i) {
#pragma HLS PIPELINE II=1
        state_output[i] = next_state[i];
    }
    if (final_tile) for (uint32_t d = 0; d < 64; ++d) {
#pragma HLS PIPELINE II=1
        output[d] = result[d];
    }
    return OK;
}
