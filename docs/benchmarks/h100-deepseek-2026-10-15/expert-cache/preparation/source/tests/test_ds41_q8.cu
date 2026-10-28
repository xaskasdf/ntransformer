// Standalone projection correctness + optional CUDA-event microbenchmark.
#include "model/ds41/ds41_q8.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t size) { check(cudaMalloc(&data, size)); }
    ~Buffer() { if (data) cudaFree(data); }
};
uint32_t random_bits(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
void run(int rows, int cols, int groups, bool bench) {
    const size_t nb = cols / 32;
    std::vector<uint8_t> weights((size_t)groups * rows * nb * 34);
    uint32_t seed = 0x12345678;
    for (size_t b = 0; b < weights.size() / 34; ++b) {
        const __half scale = __float2half_rn(std::ldexp(1.0f + (random_bits(seed) % 31) / 32.0f,
            int(random_bits(seed) % 16) - 12));
        std::memcpy(weights.data() + b * 34, &scale, 2);
        for (int i = 0; i < 32; ++i) weights[b * 34 + 2 + i] = uint8_t(random_bits(seed));
    }
    std::vector<float> x(groups * cols), reference(groups * rows), candidate(groups * rows);
    Buffer w(weights.size()), input(x.size() * 4), output(reference.size() * 4);
    check(cudaMemcpy(w.data, weights.data(), weights.size(), cudaMemcpyHostToDevice));
    auto launch = [&](bool cooperative) {
        if (groups > 1) nt::ds41::detail::launch_q8_projection<true>(
            (const uint8_t*)w.data, (const float*)input.data, (float*)output.data,
            rows, cols, groups, cooperative, nullptr);
        else nt::ds41::detail::launch_q8_projection(
            (const uint8_t*)w.data, (const float*)input.data, (float*)output.data,
            rows, cols, 1, cooperative, nullptr);
    };
    for (int round = 0; round < 3; ++round) {
        for (float& v : x) v = round == 0 ? 0.0f :
            (int(random_bits(seed) % 20001) - 10000) * (round == 1 ? 1e-4f : 1.37f);
        check(cudaMemcpy(input.data, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        launch(false);
        check(cudaMemcpy(reference.data(), output.data, reference.size() * 4, cudaMemcpyDeviceToHost));
        launch(true);
        check(cudaMemcpy(candidate.data(), output.data, candidate.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(reference.data(), candidate.data(), reference.size() * 4))
            throw std::runtime_error("cooperative output differs from reference kernel");
        // Independent high precision CPU dot catches common addressing bugs.
        // Bound error by sum of magnitudes, including cancellation-heavy rows.
        for (int g = 0; g < groups; ++g) for (int r = 0; r < rows; ++r) {
            double expected = 0, magnitude = 0;
            for (int c = 0; c < cols; ++c) {
                const auto* block = weights.data() + (((size_t)g * rows + r) * nb + c / 32) * 34;
                __half scale; std::memcpy(&scale, block, 2);
                const double term = double(__half2float(scale)) * int(int8_t(block[2 + c % 32])) * x[g * cols + c];
                expected += term; magnitude += std::fabs(term);
            }
            const float actual = candidate[g * rows + r];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 2e-6 * magnitude + 1e-6)
                throw std::runtime_error("projection differs from independent CPU dot");
        }
    }
    std::printf("PASS Q8 rows=%d cols=%d groups=%d bitwise=1 cooperative=%d\n",
        rows, cols, groups, nt::ds41::detail::q8_cooperative_supported(cols));
    if (bench) {
        cudaEvent_t start, end;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        std::vector<float> times[2];
        for (int repetition = 0; repetition < 7; ++repetition) for (int i = 0; i < 2; ++i) {
            const int variant = (i + repetition) % 2;
            for (int warm = 0; warm < 5; ++warm) launch(variant);
            check(cudaEventRecord(start));
            for (int step = 0; step < 20; ++step) launch(variant);
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, start, end));
            times[variant].push_back(ms / 20);
        }
        for (auto& t : times) std::sort(t.begin(), t.end());
        std::printf("BENCH rows=%d cols=%d groups=%d legacy_ms=%.9g cooperative_ms=%.9g speedup=%.5f\n",
            rows, cols, groups, times[0][3], times[1][3], times[0][3] / times[1][3]);
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
}
}
int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: CUDA unavailable"); return 77;
    }
    try {
        const bool bench = argc == 2 && std::string(argv[1]) == "bench";
        if (argc > 2 || (argc == 2 && !bench)) throw std::runtime_error("expected [bench]");
        for (int cols : {32, 64, 96, 1088, 1280, 5120, 8192}) run(9, cols, 1, false);
        run(3, 1088, 3, false);
        run(9, 96, 3, false);
        if (bench) {
            run(1280, 5120, 1, true); // Q-A
            run(32768, 1280, 1, true); // Q-B
            run(512, 5120, 1, true); // KV
            run(1024, 4096, 8, true); // grouped O-A
            run(5120, 8192, 1, true); // O-B
        }
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
