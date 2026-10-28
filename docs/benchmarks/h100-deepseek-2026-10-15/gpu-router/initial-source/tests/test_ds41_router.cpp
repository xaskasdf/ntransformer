#include "model/ds41/ds41_gpu.h"
#include "model/ds41/ds41_model.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
struct Fixture {
    int rows, cols;
    std::vector<float> f32;
    std::vector<uint16_t> bf16;
    Fixture(int r, int c) : rows(r), cols(c), f32(size_t(r) * c), bf16(f32.size()) {
        for (size_t i = 0; i < f32.size(); ++i) {
            float v = std::sin(float(i % 131071) * .013f) * .02f;
            f32[i] = v;
            uint32_t bits; std::memcpy(&bits, &v, 4); bf16[i] = uint16_t(bits >> 16);
        }
    }
};
}

int main(int argc, char** argv) {
    try {
        const bool disabled = argc > 1 && std::string(argv[1]) == "disabled";
        if (disabled) {
            float x = 1, out = -123;
            require(!nt::ds41::gpu_router(&x, &x, &out, 1, 1, false, false), "disabled router succeeded");
            require(out == -123, "disabled router published output");
            std::puts("PASS disabled router"); return 0;
        }
        if (!nt::ds41::gpu_backbone_enabled()) { std::puts("SKIP CUDA unavailable"); return 77; }
#if defined(__AVX2__)
        constexpr bool cpu_avx2 = true;
#else
        constexpr bool cpu_avx2 = false;
#endif
        // Keep every immutable weight allocation alive for the resident registry.
        Fixture cases[] = {{7, 19}, {384, 5120}, {257, 8192}, {1, 1}};
        nt::ds41::Model model;
        uint64_t calls = 0;
        const auto before = nt::ds41::gpu_backbone_stats();
        for (const auto& c : cases) for (bool bf16 : {false, true}) for (int round = 0; round < 3; ++round) {
            std::vector<float> x(c.cols), expected(c.rows), actual(c.rows, -123);
            for (int i = 0; i < c.cols; ++i) x[i] = std::cos(float(i + 1) * (round + 1) * .031f);
            nt::ds41::TensorRef w;
            w.data = bf16 ? reinterpret_cast<const uint8_t*>(c.bf16.data()) : reinterpret_cast<const uint8_t*>(c.f32.data());
            w.dtype = bf16 ? nt::DType::BF16 : nt::DType::F32;
            w.ne[0] = c.cols; w.ne[1] = c.rows;
            model.matvec(w, x.data(), expected.data());
            require(nt::ds41::gpu_router(w.data, x.data(), actual.data(), c.rows, c.cols, bf16, cpu_avx2), "router unavailable");
            ++calls;
            for (int i = 0; i < c.rows; ++i) {
                require(std::isfinite(actual[i]), "nonfinite output");
                if (std::memcmp(&actual[i], &expected[i], 4)) {
                    std::fprintf(stderr, "shape=%dx%d bf16=%d row=%d actual=%.9g expected=%.9g\n",
                        c.rows, c.cols, bf16, i, actual[i], expected[i]);
                    throw std::runtime_error("CPU/GPU projection differs");
                }
            }
            // Exercise the scalar portability path independently of this CPU build.
            for (int row = 0; row < c.rows; ++row) {
                float acc = 0;
                for (int col = 0; col < c.cols; ++col) {
                    const size_t index = (size_t)row * c.cols + col;
                    float weight = c.f32[index];
                    if (bf16) { uint32_t bits = uint32_t(c.bf16[index]) << 16; std::memcpy(&weight, &bits, 4); }
                    acc = std::fma(weight, x[col], acc);
                }
                expected[row] = acc;
            }
            require(nt::ds41::gpu_router(w.data, x.data(), actual.data(), c.rows, c.cols, bf16, false), "scalar router unavailable");
            ++calls;
            require(std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0, "scalar projection differs");
        }
        float x = 1, out = -123;
        require(!nt::ds41::gpu_router(&x, &x, &out, 4097, 1, false, true), "bad rows accepted");
        require(!nt::ds41::gpu_router(&x, &x, &out, 1, 65537, false, true), "bad columns accepted");
        require(out == -123, "invalid shape published output");
        require(nt::ds41::gpu_backbone_stats().router_calls - before.router_calls == calls, "GPU coverage mismatch");
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA error");
        std::printf("PASS router: %llu exact F32/BF16 projections, AVX2/scalar order, tails, buffer growth and bounds\n", (unsigned long long)calls);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
