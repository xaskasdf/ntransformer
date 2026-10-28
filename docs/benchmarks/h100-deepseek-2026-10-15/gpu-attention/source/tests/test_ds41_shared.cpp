#include "model/ds41/ds41_gpu.h"
#include "model/ds41/ds41_model.h"
#include "model/ds41/ds41_shared.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ds41;
namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Device() { cudaFree(p); }
    float* f() { return static_cast<float*>(p); }
    void put(const void* src, size_t bytes) { check(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice)); }
    std::vector<float> get(int n) {
        std::vector<float> result(n);
        check(cudaMemcpy(result.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }
};
struct Matrix {
    std::vector<uint8_t> data;
    TensorRef tensor;
    Matrix(int rows, int cols, unsigned seed) : data((size_t)rows * (cols / 32) * 34) {
        uint32_t rng = seed;
        for (size_t b = 0; b < data.size() / 34; ++b) {
            uint16_t scale = uint16_t(0x1800 + ((b * 13 + seed) % 2048));
            std::memcpy(data.data() + b * 34, &scale, 2);
            for (int j = 0; j < 32; ++j) {
                rng = rng * 1664525u + 1013904223u;
                data[b * 34 + 2 + j] = uint8_t(rng >> 24);
            }
        }
        tensor.data = data.data(); tensor.dtype = nt::DType::Q8_0;
        tensor.ne[0] = cols; tensor.ne[1] = rows; tensor.ne[2] = tensor.ne[3] = 1;
    }
};
void exact(const char* label, const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite projection");
        if (std::memcmp(&a[i], &b[i], 4)) {
            std::fprintf(stderr, "%s[%zu]: %.9g != %.9g\n", label, i, a[i], b[i]);
            throw std::runtime_error("projection not bitwise equal");
        }
    }
}
void close(const char* label, const std::vector<float>& a, const std::vector<float>& b,
           float atol, float rtol, bool normwise = false) {
    double error2 = 0, ref2 = 0;
    float maximum = 0;
    size_t changed = 0;
    size_t outside = 0;
    float ref_max = 0;
    for (float v : b) ref_max = std::max(ref_max, std::fabs(v));
    for (size_t i = 0; i < a.size(); ++i) {
        float error = std::fabs(a[i] - b[i]);
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite shared output");
        const float bound = atol + rtol * std::fabs(b[i]);
        if (error > bound) {
            if (outside == 0) std::printf("first outside %s[%zu]: %.9g vs %.9g, error %.9g, limit %.9g\n",
                label, i, a[i], b[i], error, bound);
            ++outside;
        }
        maximum = std::max(maximum, error);
        error2 += double(error) * error; ref2 += double(b[i]) * b[i];
        changed += std::memcmp(&a[i], &b[i], 4) != 0;
    }
    double rel_l2 = std::sqrt(error2 / std::max(ref2, 1e-30));
    std::printf("%s max_abs=%.9g rel_l2=%.9g ref_max=%.9g changed=%zu/%zu\n", label, maximum,
        rel_l2, ref_max, changed, a.size());
    if (normwise && ref2 > 1e-20) require(rel_l2 <= rtol, "shared normwise error exceeded");
    require(outside == 0, "shared output tolerance exceeded");
}
}

int main(int argc, char** argv) {
    try {
#if !defined(__AVX2__)
        std::puts("SKIP CPU AVX2 reference unavailable"); return 77;
#endif
        std::string shape = argc > 1 ? argv[1] : "small";
        int h = shape == "max" ? 8192 : shape == "real" ? 5120 : 96;
        int ff = shape == "max" ? 8192 : shape == "real" ? 2304 : 64;
        require(shape == "small" || shape == "real" || shape == "max" || shape == "disabled", "unknown shape");
        Matrix gate(ff, h, 1), up(ff, h, 2), down(h, ff, 3);
        SharedExpert shared{gate.data.data(), up.data.data(), down.data.data(), ff, 2.0f};
        std::vector<float> x(h, 1.0f), out(h, -123.0f);
        if (shape == "disabled") {
            require(!gpu_shared_expert(x.data(), h, shared, out.data()), "disabled shared GPU succeeded");
            for (float v : out) require(v == -123.0f, "disabled path changed output");
            std::puts("PASS disabled shared GPU"); return 0;
        }
        if (!gpu_experts_enabled()) { std::puts("SKIP CUDA unavailable"); return 77; }
        Model model;
        Device dg(gate.data.size()), du(up.data.size()), dd(down.data.size());
        dg.put(gate.data.data(), gate.data.size()); du.put(up.data.data(), up.data.size());
        dd.put(down.data.data(), down.data.size());
        Device dx(h * 4), dgate(ff * 4), dup(ff * 4), dh(ff * 4), dy(h * 4);
        std::vector<float> cg(ff), cu(ff), ch(ff), cy(h), cy_gpu_h(h), dumps;
        for (int round = 0; round < 4; ++round) {
            for (int i = 0; i < h; ++i) x[i] = std::sin(float(i + 3) * (round + 1) * 0.137f) * (round == 3 ? 8.0f : 1.0f);
            x[0] = -0.0f;
            shared.limit = round == 0 ? 0.0f : round == 1 ? 1e-6f : 2.0f;
            model.matvec(gate.tensor, x.data(), cg.data());
            model.matvec(up.tensor, x.data(), cu.data());
            dx.put(x.data(), x.size() * 4);
            detail::launch_shared_q8(dg.p, du.p, dx.f(), dgate.f(), dup.f(), ff, h, true, nullptr);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            exact("gate", dgate.get(ff), cg); exact("up", dup.get(ff), cu);
            for (int i = 0; i < ff; ++i) {
                float g = cg[i], u = cu[i];
                if (shared.limit > 1e-6f) {
                    g = std::min(g, shared.limit);
                    u = std::max(-shared.limit, std::min(u, shared.limit));
                }
                ch[i] = (g / (1.0f + expf(-g))) * u;
            }
            detail::launch_shared_activation(dgate.f(), dup.f(), dh.f(), ff, shared.limit, nullptr);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            auto gh = dh.get(ff);
            close("activation", gh, ch, 2e-6f, 1e-6f);
            model.matvec(down.tensor, gh.data(), cy_gpu_h.data());
            detail::launch_shared_q8(dd.p, nullptr, dh.f(), dy.f(), nullptr, h, ff, true, nullptr);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            exact("down with identical activation", dy.get(h), cy_gpu_h);
            model.matvec(down.tensor, ch.data(), cy.data());
            require(gpu_shared_expert(x.data(), h, shared, out.data()), "shared helper unavailable");
            exact("helper vs component kernels", out, cy_gpu_h);
            // Check both pointwise error and relative L2 after the activation;
            // projections with identical inputs are checked bitwise above.
            close("shared FFN", out, cy, 3e-4f, 1e-6f, true);
            dumps.insert(dumps.end(), out.begin(), out.end());
        }
        // Integrated path must preserve routed output exactly and share neither
        // scratch nor output storage with the independent shared result.
        int routed_ff = std::min(ff, 4096);
        size_t slice = (size_t)h * routed_ff / 32 * 17;
        std::vector<uint8_t> rg(6 * slice), ru(rg.size()), rd(rg.size());
        for (auto* w : {&rg, &ru, &rd}) for (size_t b = 0; b < w->size() / 17; ++b) {
            (*w)[b * 17] = 120;
            for (int i = 1; i < 17; ++i) (*w)[b * 17 + i] = uint8_t(b * 7 + i * 19);
        }
        int ids[6] = {5, 0, 3, 1, 4, 2}; float weights[6] = {.1f, .2f, .3f, .1f, .2f, .1f};
        std::vector<float> routed(h), routed_ref(h), shared_out(h);
        require(gpu_moe_experts(x.data(), h, rg.data(), ru.data(), rd.data(), routed_ff,
            ids, weights, 6, 2.0f, routed_ref.data()), "routed reference unavailable");
        require(gpu_moe_experts_shared(x.data(), h, rg.data(), ru.data(), rd.data(), routed_ff,
            ids, weights, 6, 2.0f, routed.data(), shared, shared_out.data()), "joint path unavailable");
        exact("joint routed", routed, routed_ref); exact("joint shared", shared_out, out);
        auto stats = gpu_moe_stats();
        require(stats.shared_calls == 1 && stats.shared_kernel_launches == 3, "shared coverage mismatch");
        const char* early_flag = std::getenv("DS41_SHARED_EARLY");
        const bool early = !early_flag || std::atoi(early_flag) != 0;
        require(stats.shared_early_calls == (early ? 1u : 0u), "shared scheduling mode mismatch");
        require(stats.shared_weight_bytes == gate.data.size() * 3, "shared residency changed");
        shared.ff = 63;
        std::fill(out.begin(), out.end(), -123.0f);
        require(!gpu_shared_expert(x.data(), h, shared, out.data()), "invalid shared shape accepted");
        for (float v : out) require(v == -123.0f, "invalid shared shape published output");
        if (argc > 2) {
            FILE* f = std::fopen(argv[2], "wb"); require(f != nullptr, "cannot open output");
            size_t n = std::fwrite(dumps.data(), 4, dumps.size(), f);
            int closed = std::fclose(f); require(n == dumps.size() && closed == 0, "short output");
        }
        std::puts("PASS shared GPU: exact Q8 reductions, bounded activation error, integrated outputs, residency, bounds");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
