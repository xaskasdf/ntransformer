#pragma once

#include "ds41_model.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace nt::ds41::detail {

// Input and output follow GGUF/ggml's [dst, src] layout: dst varies fastest.
// The temporary uses rows indexed by dst to preserve the reduction order.
inline void hc_sinkhorn(const float* logits, const float* base, float scale,
                        float* comb, int hc, float eps, int iterations) {
    NT_CHECK(hc > 0 && hc <= 8 && iterations >= 1, "invalid HC Sinkhorn dimensions");
    float matrix[64];
    for (int dst = 0; dst < hc; ++dst) {
        for (int src = 0; src < hc; ++src) {
            matrix[dst * hc + src] = logits[dst + src * hc] * scale + base[dst + src * hc];
        }
    }
    for (int src = 0; src < hc; ++src) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int dst = 0; dst < hc; ++dst) maximum = std::max(maximum, matrix[dst * hc + src]);
        float sum = 0.0f;
        for (int dst = 0; dst < hc; ++dst) {
            matrix[dst * hc + src] = expf(matrix[dst * hc + src] - maximum);
            sum += matrix[dst * hc + src];
        }
        for (int dst = 0; dst < hc; ++dst) matrix[dst * hc + src] /= sum;
    }
    for (int i = 0; i < hc * hc; ++i) matrix[i] += eps;
    auto norm_cols = [&]() {
        for (int dst = 0; dst < hc; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < hc; ++src) sum += matrix[dst * hc + src];
            sum += eps;
            for (int src = 0; src < hc; ++src) matrix[dst * hc + src] /= sum;
        }
    };
    auto norm_rows = [&]() {
        for (int src = 0; src < hc; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < hc; ++dst) sum += matrix[dst * hc + src];
            sum += eps;
            for (int dst = 0; dst < hc; ++dst) matrix[dst * hc + src] /= sum;
        }
    };
    norm_cols();
    for (int i = 1; i < iterations; ++i) { norm_rows(); norm_cols(); }
    // Export dst-fast storage for hc_post_mix. A flat copy would transpose
    // the directed residual connections, while preserving row/column sums.
    for (int dst = 0; dst < hc; ++dst)
        for (int src = 0; src < hc; ++src)
            comb[dst + src * hc] = matrix[dst * hc + src];
}

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
