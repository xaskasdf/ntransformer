// Small numerical regressions for attention offload, without an expert cache or
// model files. Each CTest case starts a fresh process to exercise initialization.
#include "model/ds41/ds41_gpu.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

void prologue(std::vector<float>& outputs) {
    const auto before = nt::ds41::gpu_backbone_stats();
    constexpr int hidden = 96, rank = 32, hd = 64, heads = 4, rd = 16;
    Matrix qa(rank, hidden, 1), qb(heads * hd, rank, 2), kv(hd, hidden, 3);
    std::unique_ptr<nt::ds41::GpuAttentionState, decltype(&nt::ds41::gpu_attention_destroy)> resident(
        nt::ds41::gpu_attention_create(1, 2, heads, hd), nt::ds41::gpu_attention_destroy);
    require(bool(resident), "resident prologue state unavailable");
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
        outputs.insert(outputs.end(), qr.begin(), qr.end());
        outputs.insert(outputs.end(), q.begin(), q.end());
        outputs.insert(outputs.end(), k.begin(), k.end());
        std::vector<float> qr_res(rank), q_res(heads * hd, -123), kv_res(hd), recovered(heads * hd);
        require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(), qb.q8.data(), kv.q8.data(),
            kn.data(), rank, hd, heads, rd, 1e-6f, cs.data(), sn.data(), qr_res.data(), q_res.data(), kv_res.data(),
            resident.get(), 0, r), "resident prologue unavailable");
        require(std::memcmp(qr.data(), qr_res.data(), rank * 4) == 0 && std::memcmp(k.data(), kv_res.data(), hd * 4) == 0,
            "resident prologue changed CPU mirrors");
        require(std::all_of(q_res.begin(), q_res.end(), [](float v) { return v == -123; }), "resident Q was read back");
        require(nt::ds41::gpu_attention_read_query(resident.get(), recovered.data()), "resident Q recovery unavailable");
        require(std::memcmp(q.data(), recovered.data(), q.size() * 4) == 0, "resident Q differs");
    }
    const char* flag = std::getenv("DS41_QKV_READBACK_BATCH");
    const bool batched = !flag || std::atoi(flag) != 0;
    const auto after = nt::ds41::gpu_backbone_stats();
    require(after.prologue_calls - before.prologue_calls == 6, "prologue coverage incomplete");
    require(after.prologue_readback_waits - before.prologue_readback_waits == (batched ? 6u : 15u),
            "unexpected prologue readback waits");
    require(after.prologue_readback_bytes - before.prologue_readback_bytes == 3u * (2 * rank + heads * hd + 2 * hd) * 4,
            "unexpected prologue transfer bytes");

    // Selective mirrors must preserve the GPU arithmetic and all prior ring
    // slots, including after a failed tail, wrap, reset, and input reuse.
    constexpr int steps = 9, window = 3;
    std::vector<float> queries[steps], latents[steps], keys[steps];
    for (int t = 0; t < steps; ++t) {
        auto x = input(hidden, t);
        queries[t].resize(heads * hd); latents[t].resize(rank); keys[t].resize(hd);
        require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(), qb.q8.data(), kv.q8.data(),
            kn.data(), rank, hd, heads, rd, 1e-6f, cs.data(), sn.data(), latents[t].data(), queries[t].data(), keys[t].data()),
            "selective reference failed");
    }
    decltype(resident) states[] = {
        {nt::ds41::gpu_attention_create(2, window, heads, hd), nt::ds41::gpu_attention_destroy},
        {nt::ds41::gpu_attention_create(2, window, heads, hd), nt::ds41::gpu_attention_destroy}};
    require(states[0] && states[1], "selective states unavailable");
    const auto selective_before = nt::ds41::gpu_backbone_stats();
    uint64_t expected_bytes = 0, expected_waits = 0, deferred = 0;
    for (int epoch = 0; epoch < 2; ++epoch) {
        for (auto& state : states) nt::ds41::gpu_attention_reset(state.get());
        for (int t = 0; t < steps; ++t) for (int owner = 0; owner < 2; ++owner) for (int layer = 0; layer < 2; ++layer) {
            const int sample = (t + 3 * owner + layer) % steps;
            auto x = input(hidden, sample), cos_copy = cs, sin_copy = sn;
            std::vector<float> qr_mirror(rank, -123), kv_mirror(hd, -123), q_recovered(heads * hd);
            const bool read_qr = (t + layer + owner) % 4 & 1;
            const bool read_kv = (t + layer + owner) % 4 & 2;
            auto* state = states[owner].get();
            require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(), qb.q8.data(), kv.q8.data(),
                kn.data(), rank, hd, heads, rd, 1e-6f, cos_copy.data(), sin_copy.data(),
                read_qr ? qr_mirror.data() : nullptr, nullptr, read_kv ? kv_mirror.data() : nullptr, state, layer, t),
                "selective prologue failed");
            expected_bytes += (read_qr * rank + read_kv * hd) * sizeof(float);
            expected_waits += batched ? (read_qr || read_kv) : read_qr + read_kv;
            deferred += !read_qr && !read_kv;
            // These buffers belong to the caller, while the following BF16
            // projection immediately reuses global pinned and device scratch.
            std::fill(x.begin(), x.end(), -37);
            std::fill(cos_copy.begin(), cos_copy.end(), -19);
            std::fill(sin_copy.begin(), sin_copy.end(), 23);
            std::vector<float> scratch(hd);
            require(nt::ds41::gpu_comp_state(x.data(), hidden, hd, kv.bf16.data(), nullptr, scratch.data(), nullptr),
                "intervening compressor failed");
            if (read_qr) require(std::memcmp(qr_mirror.data(), latents[sample].data(), rank * 4) == 0, "selective QR changed");
            if (read_kv) require(std::memcmp(kv_mirror.data(), keys[sample].data(), hd * 4) == 0, "selective KV changed");
            std::vector<float> sinks(heads), out(hidden, -123);
            require(!nt::ds41::gpu_attention_tail(state, layer, t, {}, sinks.data(), qa.q8.data(), qb.q8.data(),
                0, rank, hidden, out.data()), "unsupported tail accepted");
            require(std::all_of(out.begin(), out.end(), [](float f) { return f == -123; }), "failed tail published output");
            require(nt::ds41::gpu_attention_read_query(state, q_recovered.data()), "selective Q recovery failed");
            require(std::memcmp(q_recovered.data(), queries[sample].data(), heads * hd * 4) == 0, "selective Q changed");
            std::vector<float> ring(window * hd + 1, -123);
            require(!nt::ds41::gpu_attention_read_raw(state, layer, t, ring.data()), "stale recovery accepted");
            require(std::all_of(ring.begin(), ring.end(), [](float f) { return f == -123; }), "stale recovery published output");
            require(nt::ds41::gpu_attention_read_raw(state, layer, t + 1, ring.data()), "ring recovery failed");
            for (int j = std::max(0, t - window + 1); j <= t; ++j)
                require(std::memcmp(ring.data() + (j % window) * hd, keys[(j + 3 * owner + layer) % steps].data(), hd * 4) == 0,
                    "ring history changed or crossed a layer/session boundary");
            for (int j = std::min(window, t + 1) * hd; j <= window * hd; ++j)
                require(ring[j] == -123, "recovery touched unwritten ring slots or guard");
        }
    }
    const auto selective_after = nt::ds41::gpu_backbone_stats();
    require(selective_after.prologue_readback_bytes - selective_before.prologue_readback_bytes == expected_bytes,
            "selective readback byte count");
    require(selective_after.prologue_readback_waits - selective_before.prologue_readback_waits == expected_waits,
            "selective readback wait count");
    require(selective_after.prologue_deferred_calls - selective_before.prologue_deferred_calls == deferred,
            "selective deferred count");
    // Reuse a state's upload storage without any intervening completion wait,
    // then reset and destroy it with a final submission still outstanding.
    nt::ds41::gpu_attention_reset(states[0].get());
    for (int t = 0; t < steps; ++t) {
        auto x = input(hidden, t), cos_copy = cs, sin_copy = sn;
        require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(), qb.q8.data(), kv.q8.data(),
            kn.data(), rank, hd, heads, rd, 1e-6f, cos_copy.data(), sin_copy.data(), nullptr, nullptr, nullptr,
            states[0].get(), 0, t), "consecutive deferred prologue failed");
        std::fill(x.begin(), x.end(), -37);
        std::fill(cos_copy.begin(), cos_copy.end(), -19);
        std::fill(sin_copy.begin(), sin_copy.end(), 23);
    }
    std::vector<float> ring(window * hd), recovered(heads * hd);
    require(nt::ds41::gpu_attention_read_raw(states[0].get(), 0, steps, ring.data()), "queued ring recovery failed");
    for (int t = steps - window; t < steps; ++t)
        require(std::memcmp(ring.data() + t % window * hd, keys[t].data(), hd * 4) == 0, "queued ring changed");
    require(nt::ds41::gpu_attention_read_query(states[0].get(), recovered.data()), "queued query recovery failed");
    require(std::memcmp(recovered.data(), queries[steps - 1].data(), heads * hd * 4) == 0, "queued query changed");
    for (int epoch = 0; epoch < 2; ++epoch) {
        nt::ds41::gpu_attention_reset(states[0].get());
        auto x = input(hidden, epoch);
        require(nt::ds41::gpu_attn_prologue(x.data(), hidden, qa.q8.data(), qn.data(), qb.q8.data(), kv.q8.data(),
            kn.data(), rank, hd, heads, rd, 1e-6f, cs.data(), sn.data(), nullptr, nullptr, nullptr, states[0].get(), 0, 0),
            "reset with pending upload failed");
    }
    std::puts("PASS selective mirrors: exact Q/QR/KV, input reuse, ring history, fallback, reset and isolated owners");
}

void tail(std::vector<float>& outputs) {
    const auto before = nt::ds41::gpu_backbone_stats();
    struct Case {
        int hidden, hd, heads, groups, rank;
        Matrix a, b;
        Case(int h, int d, int n, int g, int r) : hidden(h), hd(d), heads(n), groups(g), rank(r),
            a(g * r, n / g * d, 4), b(h, g * r, 5) {}
    };
    // Keep all weight allocations alive: the registry keys tensors by address.
    // Eight groups with rank 36 also exercises a partially occupied final block.
    Case cases[] = {{96, 64, 4, 2, 32}, {128, 64, 8, 8, 36},
                    {64, 32, 1, 1, 64}, {96, 64, 6, 3, 64}};
    for (const auto& c : cases) {
        const int gdim = c.heads / c.groups * c.hd;
        for (int r = 0; r < 3; ++r) {
            const auto x = input(c.heads * c.hd, r);
            std::vector<float> latent(c.groups * c.rank), out(c.hidden);
            for (int g = 0; g < c.groups; ++g) {
                const auto part = c.a.mv(x.data() + g * gdim, g * c.rank, c.rank);
                std::copy(part.begin(), part.end(), latent.begin() + g * c.rank);
            }
            require(nt::ds41::gpu_attn_tail(x.data(), c.hd, c.heads, c.a.q8.data(), c.b.q8.data(),
                c.groups, c.rank, c.hidden, out.data()), "tail unavailable");
            close("grouped output", out, c.b.mv(latent.data()));
            outputs.insert(outputs.end(), out.begin(), out.end());
        }
    }
    const auto after = nt::ds41::gpu_backbone_stats();
    const char* value = std::getenv("DS41_GROUPED_TAIL");
    const bool grouped = !value || std::atoi(value) != 0;
    require(after.tail_calls - before.tail_calls == 12, "tail call count mismatch");
    require(after.grouped_tail_calls - before.grouped_tail_calls == (grouped ? 12 : 0),
            "grouped tail count mismatch");
    require(after.tail_gemv_launches - before.tail_gemv_launches == (grouped ? 24 : 54),
            "tail launch count mismatch");
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
        require(argc == 2 || argc == 3, "expected prologue, tail, compressor or indexer [output.bin]");
        const std::string mode = argv[1];
        require(argc == 2 || mode == "tail" || mode == "prologue", "output dump requires tail or prologue");
        std::vector<float> outputs;
        if (mode == "prologue") prologue(outputs);
        else if (mode == "tail") tail(outputs);
        else if (mode == "compressor") compressor();
        else if (mode == "indexer") indexer();
        else throw std::runtime_error("unknown mode");
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution error");
        if (argc == 3) {
            FILE* file = std::fopen(argv[2], "wb");
            require(file != nullptr, "cannot open output dump");
            const size_t wrote = std::fwrite(outputs.data(), sizeof(float), outputs.size(), file);
            const int closed = std::fclose(file);
            require(wrote == outputs.size() && closed == 0, "short output dump");
        }
        std::printf("PASS %s (fresh backbone, no expert cache)\n", mode.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
