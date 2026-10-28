#include "model/ds41/ds41_hc.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ds41;

static void check_sinkhorn_layout() {
    // A non-symmetric doubly stochastic matrix is unchanged by Sinkhorn
    // (eps=0). Basis-vector residuals expose a producer/consumer transpose;
    // checking only row/column sums or direct-vs-pool equivalence cannot.
    const float expected[4][4] = {
        {0.4f, 0.3f, 0.2f, 0.1f},
        {0.1f, 0.4f, 0.3f, 0.2f},
        {0.2f, 0.1f, 0.4f, 0.3f},
        {0.3f, 0.2f, 0.1f, 0.4f},
    };
    float logits[16], base[16] = {}, comb[16], residual[16] = {};
    float branch[4] = {}, post[4] = {}, out[16];
    for (int dst = 0; dst < 4; ++dst)
        for (int src = 0; src < 4; ++src)
            logits[dst + src * 4] = logf(expected[dst][src]);
    for (int src = 0; src < 4; ++src) residual[src + src * 4] = 1.0f;
    for (int iterations : {1, 20}) {
        detail::hc_sinkhorn(logits, base, 1.0f, comb, 4, 0.0f, iterations);
        for (bool direct : {false, true}) {
            detail::hc_post_mix(branch, residual, post, comb, out, 4, 4, direct, 4);
            for (int dst = 0; dst < 4; ++dst) {
                for (int src = 0; src < 4; ++src) {
                    if (std::abs(comb[dst + src * 4] - expected[dst][src]) > 2e-6f ||
                        std::abs(out[src + dst * 4] - expected[dst][src]) > 2e-6f)
                        throw std::runtime_error("HC Sinkhorn producer/consumer transposes destination and source");
                }
            }
        }
    }
}

static void equal(const std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            std::memcmp(&a[i], &b[i], sizeof(float))) {
            throw std::runtime_error("nonfinite or unequal value at " + std::to_string(i));
        }
    }
}

static void check(int hidden, int hc, int pattern, bool benchmark) {
    std::vector<float> x((size_t)hidden * hc), branch(hidden), weights(hc), post(hc), comb(hc * hc);
    auto value = [pattern](int i) {
        float v = ((i * 17 % 127) - 63) / 64.0f;
        if (pattern == 1) v *= (i % 3 == 0 ? 1e10f : 1e-10f);
        if (pattern == 2) v = i % 2 ? -0.0f : 0.0f;
        return v;
    };
    for (size_t i = 0; i < x.size(); i++) x[i] = value((int)i);
    for (int i = 0; i < hidden; i++) branch[i] = value(i + 11);
    for (int i = 0; i < hc; i++) { weights[i] = value(i + 5); post[i] = value(i + 7); }
    for (int i = 0; i < hc * hc; i++) comb[i] = value(i + 19);
    std::vector<float> collapse(hidden + 2, 123.0f), collapse_pool(collapse), collapse_ref(collapse);
    std::vector<float> mix((size_t)hidden * hc + 2, 123.0f), mix_pool(mix), mix_ref(mix);

    // Independent scalar oracle, including guard elements on both sides.
    for (int d = 0; d < hidden; d++) {
        float v = 0;
        for (int h = 0; h < hc; h++) v += x[d + (size_t)h * hidden] * weights[h];
        collapse_ref[d + 1] = v;
    }
    for (int dst = 0; dst < hc; dst++) {
        for (int d = 0; d < hidden; d++) {
            float v = branch[d] * post[dst];
            for (int src = 0; src < hc; src++) v += x[d + (size_t)src * hidden] * comb[dst + src * hc];
            mix_ref[1 + d + (size_t)dst * hidden] = v;
        }
    }
    auto collapse_op = [&](bool direct) {
        detail::hc_collapse(x.data(), weights.data(), (direct ? collapse : collapse_pool).data() + 1,
                            hidden, hc, direct, 52);
    };
    auto mix_op = [&](bool direct) {
        detail::hc_post_mix(branch.data(), x.data(), post.data(), comb.data(),
                            (direct ? mix : mix_pool).data() + 1, hidden, hc, direct, 52);
    };
    collapse_op(true); collapse_op(false); mix_op(true); mix_op(false);
    equal(collapse, collapse_ref); equal(collapse_pool, collapse_ref);
    equal(mix, mix_ref); equal(mix_pool, mix_ref);
    if (benchmark) {
        // ABBA order, warm operations, no allocation inside measured loops.
        for (int round = 0; round < 4; round++) {
            bool direct = round == 1 || round == 2;
            for (int op = 0; op < 2; op++) {
                const int repeats = 400;
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < repeats; i++) {
                    if (op == 0) collapse_op(direct); else mix_op(direct);
                    // Keep repeated identical stores observable to the compiler;
                    // this is a compiler barrier, not a hardware memory fence.
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                std::printf("{\"round\":%d,\"direct\":%s,\"operation\":\"%s\",\"repeats\":%d,\"seconds\":%.9f}\n",
                            round, direct ? "true" : "false", op == 0 ? "collapse" : "post_mix", repeats, seconds);
            }
        }
        equal(collapse, collapse_pool); equal(mix, mix_pool);
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            check(5120, 4, 0, true);
        } else if (argc == 1) {
            check_sinkhorn_layout();
            for (int hidden : {0, 1, 31, 65, 256, 5120, 8193})
                for (int hc : {1, 2, 4, 8})
                    for (int pattern = 0; pattern < 3; pattern++) check(hidden, hc, pattern, false);
        } else {
            throw std::runtime_error("usage: test_ds41_hc [--benchmark]");
        }
        std::puts("PASS HC direct/pool/scalar bitwise equality and output guards");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
