// Small numerical regressions for attention offload, without an expert cache or
// model files. Each CTest case starts a fresh process to exercise initialization.
#include "model/ds41/ds41_gpu.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Matrix {
    int rows, cols;
    std::vector<uint8_t> q8;
    std::vector<uint16_t> bf16;
    std::vector<float> values;

    Matrix(int r, int c, int seed) : rows(r), cols(c), values(r * c) {
        q8.resize((size_t)r * (c / 32) * 34);
        bf16.resize((size_t)r * c);
        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < c; ++col) {
                const int v = ((row * 7 + col * 11 + seed) % 15) - 7;
                const float f = (float)v / 32.0f;
                values[row * c + col] = f;
                uint32_t bits;
                std::memcpy(&bits, &f, sizeof(bits));
                bf16[row * c + col] = (uint16_t)(bits >> 16);
                const size_t block = ((size_t)row * (c / 32) + col / 32) * 34;
                const uint16_t scale = 0x2800; // FP16 1/32
                std::memcpy(q8.data() + block, &scale, sizeof(scale));
                q8[block + 2 + col % 32] = (uint8_t)(int8_t)v;
            }
        }
    }

    std::vector<float> mv(const float* x, int first = 0, int count = -1) const {
        if (count < 0) count = rows;
        std::vector<float> y(count, 0.0f);
        for (int r = 0; r < count; ++r) {
            for (int c = 0; c < cols; ++c) y[r] += values[(first + r) * cols + c] * x[c];
        }
        return y;
    }
};

std::vector<float> input(int n, int round) {
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = std::sin((i + 3) * (round + 1) * 0.137f);
    return x;
}

void require(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}

void close(const char* label, const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "shape mismatch");
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite result");
        maximum = std::max(maximum, error);
        if (error > 2e-4f + 5e-5f * std::fabs(expected[i])) {
            std::fprintf(stderr, "FAIL %s[%zu]: %.9g vs %.9g\n", label, i, actual[i], expected[i]);
            throw std::runtime_error("numerical mismatch");
        }
    }
    std::printf("PASS %s max_abs=%.9g\n", label, maximum);
}

void norm(std::vector<float>& x, const std::vector<float>& weights) {
    float sum = 0.0f;
    for (float v : x) sum += v * v;
    const float scale = 1.0f / std::sqrt(sum / x.size() + 1e-6f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * scale * weights[i];
}

void rope(std::vector<float>& x, int heads, int hd, const std::vector<float>& cs,
          const std::vector<float>& sn) {
    for (int h = 0; h < heads; ++h) {
        for (int p = 0; p < (int)cs.size(); ++p) {
            const int i = h * hd + hd - (int)cs.size() * 2 + 2 * p;
            const float a = x[i], b = x[i + 1];
            x[i] = a * cs[p] - b * sn[p];
            x[i + 1] = a * sn[p] + b * cs[p];
        }
    }
}

void prologue() {
    constexpr int hidden = 96, rank = 32, hd = 64, heads = 4, rd = 16;
    Matrix qa(rank, hidden, 1), qb(heads * hd, rank, 2), kv(hd, hidden, 3);
    std::vector<float> qn(rank, 1.1f), kn(hd, 0.9f), cs(rd / 2), sn(rd / 2);
    for (int i = 0; i < rd / 2; ++i) {
        cs[i] = std::cos(0.3f * (i + 1));
        sn[i] = std::sin(0.3f * (i + 1));
    }
    for (int r = 0; r < 3; ++r) {
        const auto x = input(hidden, r);
        auto qr_ref = qa.mv(x.data());
        norm(qr_ref, qn);
        auto q_ref = qb.mv(qr_ref.data());
        auto kv_ref = kv.mv(x.data());
        norm(kv_ref, kn);
        rope(q_ref, heads, hd, cs, sn);
        rope(kv_ref, 1, hd, cs, sn);
        std::vector<float> qr(rank), q(heads * hd), k(hd);
        require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(),
            qb.q8.data(), kv.q8.data(), kn.data(), rank, hd, heads, rd, 1e-6f,
            cs.data(), sn.data(), qr.data(), q.data(), k.data()), "prologue unavailable");
        close("query latent", qr, qr_ref);
        close("query + RoPE", q, q_ref);
        close("KV + RoPE", k, kv_ref);
    }
}

void tail() {
    constexpr int hidden = 96, hd = 64, heads = 4, groups = 2, rank = 32;
    constexpr int gdim = heads / groups * hd;
    Matrix a(groups * rank, gdim, 4), b(hidden, groups * rank, 5);
    for (int r = 0; r < 3; ++r) {
        const auto x = input(heads * hd, r);
        std::vector<float> latent(groups * rank), out(hidden);
        for (int g = 0; g < groups; ++g) {
            const auto part = a.mv(x.data() + g * gdim, g * rank, rank);
            std::copy(part.begin(), part.end(), latent.begin() + g * rank);
        }
        require(nt::ds41::gpu_attn_tail(x.data(), hd, heads, a.q8.data(), b.q8.data(),
            groups, rank, hidden, out.data()), "tail unavailable");
        close("grouped output", out, b.mv(latent.data()));
    }
}

void compressor() {
    constexpr int hidden = 96, hd = 64;
    Matrix kv(hd, hidden, 6), gate(hd, hidden, 7);
    for (int r = 0; r < 3; ++r) {
        const auto x = input(hidden, r);
        std::vector<float> out(hd), score(hd);
        require(nt::ds41::gpu_comp_state(x.data(), hidden, hd, kv.bf16.data(),
            gate.bf16.data(), out.data(), score.data()), "compressor unavailable");
        close("compressor KV", out, kv.mv(x.data()));
        close("compressor gate", score, gate.mv(x.data()));
        require(nt::ds41::gpu_comp_state(x.data(), hidden, hd, kv.bf16.data(),
            nullptr, out.data(), nullptr), "ungated compressor unavailable");
        close("compressor ungated", out, kv.mv(x.data()));
    }
}

void indexer() {
    // Keep both matrices alive: the resident registry keys tensors by address.
    Matrix small(32, 32, 8), large(96, 256, 9);
    for (int r = 0; r < 3; ++r) {
        for (const Matrix* m : {&small, &large}) {
            const auto x = input(m->cols, r);
            std::vector<float> out(m->rows);
            for (int dtype : {0, 1}) {
                const void* w = dtype == 0 ? (const void*)m->q8.data() : m->bf16.data();
                require(nt::ds41::gpu_indexer_gemv(w, x.data(), m->cols, m->rows,
                    dtype, out.data()), "indexer unavailable");
                close(dtype ? "indexer BF16" : "indexer Q8", out, m->mv(x.data()));
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if (!nt::ds41::gpu_backbone_enabled()) {
        std::puts("SKIP: CUDA unavailable or DS41_GPU=0");
        return 77;
    }
    try {
        require(argc == 2, "expected prologue, tail, compressor or indexer");
        const std::string mode = argv[1];
        if (mode == "prologue") prologue();
        else if (mode == "tail") tail();
        else if (mode == "compressor") compressor();
        else if (mode == "indexer") indexer();
        else throw std::runtime_error("unknown mode");
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution error");
        std::printf("PASS %s (fresh backbone, no expert cache)\n", mode.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
