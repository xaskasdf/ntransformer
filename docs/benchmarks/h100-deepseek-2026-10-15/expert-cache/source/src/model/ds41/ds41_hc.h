#pragma once

#include "ds41_model.h"

namespace nt::ds41::detail {

// Small, elementwise hyper-connection operations. The direct path avoids a
// pool dispatch for only hc or hc*(hc+1) products per hidden coordinate.
// Keep the per-output accumulation order identical in both execution modes.
// Output storage must not overlap any input; dimensions come from Config.
inline void hc_collapse(const float* x, const float* weights, float* out,
                        int hidden, int hc, bool direct, int n_threads) {
    auto row = [&](int64_t d, int) {
        float acc = 0.0f;
        for (int h = 0; h < hc; h++) acc += x[d + (int64_t)h * hidden] * weights[h];
        out[d] = acc;
    };
    if (direct) {
        for (int64_t d = 0; d < hidden; d++) row(d, 0);
    } else {
        parallel_for(hidden, n_threads, row);
    }
}

inline void hc_post_mix(const float* branch, const float* residual,
                        const float* post, const float* comb, float* out,
                        int hidden, int hc, bool direct, int n_threads) {
    auto row = [&](int64_t d, int) {
        for (int dst = 0; dst < hc; dst++) {
            float v = branch[d] * post[dst];
            for (int src = 0; src < hc; src++) {
                v += residual[d + (size_t)src * hidden] * comb[dst + src * hc];
            }
            out[d + (size_t)dst * hidden] = v;
        }
    };
    if (direct) {
        for (int64_t d = 0; d < hidden; d++) row(d, 0);
    } else {
        parallel_for(hidden, n_threads, row);
    }
}

} // namespace nt::ds41::detail
