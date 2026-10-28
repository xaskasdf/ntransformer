#include "model/ds41/ds41_gpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "wide";
        require(mode == "small" || mode == "unaligned" || mode == "wide", "unknown shape");
        if (!nt::ds41::gpu_experts_enabled()) { std::puts("SKIP CUDA unavailable"); return 77; }
        const int rows = mode == "wide" ? 513 : 11;
        const int cols = mode == "wide" ? 2304 : mode == "unaligned" ? 96 : 128;
        const int blocks = cols / 32;
        const size_t slice = (size_t)rows * blocks * 17;
        std::vector<uint8_t> matrix(3 * slice);
        for (size_t b = 0; b < matrix.size() / 17; ++b) {
            matrix[b * 17] = 119 + b % 10;
            for (int j = 1; j < 17; ++j) matrix[b * 17 + j] = uint8_t(b * 73 + j * 41);
        }
        const double values[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        std::vector<float> x(cols), guarded(rows + 2), dump;
        double max_error = 0;
        for (int epoch = 0; epoch < 3; ++epoch) {
            for (int i = 0; i < cols; ++i) x[i] = std::sin(float(i + 1) * .031f * (epoch + 1));
            for (int expert : {2, 0, 1}) {
                std::fill(guarded.begin(), guarded.end(), -123.0f);
                require(nt::ds41::gpu_gemv_expert(matrix.data(), expert, rows, cols, x.data(), guarded.data() + 1), "GPU GEMV failed");
                require(guarded.front() == -123 && guarded.back() == -123, "GEMV wrote outside host output");
                for (int row = 0; row < rows; ++row) {
                    double reference = 0;
                    for (int b = 0; b < blocks; ++b) {
                        const uint8_t* q = matrix.data() + expert * slice + ((size_t)row * blocks + b) * 17;
                        const double scale = std::ldexp(1.0, int(q[0]) - 128);
                        for (int j = 0; j < 16; ++j)
                            reference += scale * (values[q[j + 1] & 15] * double(x[b * 32 + j]) +
                                                  values[q[j + 1] >> 4] * double(x[b * 32 + j + 16]));
                    }
                    const float actual = guarded[row + 1];
                    require(std::isfinite(actual), "nonfinite GEMV output");
                    const double error = std::abs(double(actual) - reference);
                    max_error = std::max(max_error, error);
                    require(error <= 2e-4 + 2e-5 * std::abs(reference), "GEMV differs from FP64 reference");
                    dump.push_back(actual);
                }
            }
        }
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution error");
        if (argc > 2) {
            FILE* file = std::fopen(argv[2], "wb");
            require(file != nullptr, "cannot open dump");
            const size_t wrote = std::fwrite(dump.data(), sizeof(float), dump.size(), file);
            const int closed = std::fclose(file);
            require(wrote == dump.size() && closed == 0, "incomplete dump");
        }
        std::printf("PASS MXFP4 rows=%d cols=%d values=%zu max_abs=%.9g; cold/hit, partial CTA/tile, word alignment fallback\n",
                    rows, cols, dump.size(), max_error);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
