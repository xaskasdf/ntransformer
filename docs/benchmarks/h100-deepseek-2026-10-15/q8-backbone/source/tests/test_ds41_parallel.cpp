// Scheduling must preserve exactly-once visits, nested calls and row arithmetic.
// This test needs no model assets and performs no CUDA operations.
#include "model/ds41/ds41_model.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        using nt::ds41::parallel_for;
        for (int repeat = 0; repeat < 5; ++repeat) {
            for (int n : {0, 1, 2, 31, 48, 63, 64, 65, 127, 129, 384, 2304, 5120, 8193}) {
                std::vector<std::atomic<int>> visits(n);
                for (auto& v : visits) v.store(0);
                parallel_for(n, 0, [&](int64_t row, int) {
                    require(row >= 0 && row < n, "out-of-range row");
                    visits[row].fetch_add(1);
                });
                for (auto& v : visits) require(v.load() == 1, "missing or repeated row");
            }
        }
        std::vector<int> nested(257 * 17, 0);
        parallel_for(257, 0, [&](int64_t outer, int) {
            parallel_for(17, 0, [&](int64_t inner, int) { ++nested[outer * 17 + inner]; });
        });
        for (int n : nested) require(n == 1, "nested dispatch failed");

        nt::ds41::Model model;
        for (auto dtype : {nt::DType::F32, nt::DType::Q8_0}) {
            constexpr int rows = 257, cols = 96;
            std::vector<float> input(cols), storage(rows * cols);
            for (int i = 0; i < cols; ++i) input[i] = std::sin(float(i) * 0.1f);
            nt::ds41::TensorRef tensor;
            tensor.dtype = dtype; tensor.ne[0] = cols; tensor.ne[1] = rows;
            std::vector<unsigned char> quantized(rows * (cols / 32) * 34);
            if (dtype == nt::DType::F32) {
                for (int i = 0; i < rows * cols; ++i) storage[i] = std::cos(float(i) * 0.03f);
                tensor.data = reinterpret_cast<const uint8_t*>(storage.data());
            } else {
                for (size_t i = 0; i < quantized.size(); i += 34) {
                    const uint16_t scale = 0x3000; // FP16 0.125
                    std::memcpy(quantized.data() + i, &scale, sizeof(scale));
                    for (int j = 0; j < 32; ++j) quantized[i + 2 + j] = (unsigned char)((i + j * 7) % 255 - 127);
                }
                tensor.data = quantized.data();
            }
            std::vector<float> expected(rows), actual(rows);
            for (int r = 0; r < rows; ++r) expected[r] = model.dot_row(tensor, r, input.data());
            model.matvec(tensor, input.data(), actual.data());
            for (float v : actual) require(std::isfinite(v), "nonfinite dot product");
            require(std::memcmp(actual.data(), expected.data(), rows * sizeof(float)) == 0,
                    "row scheduling changed dot products");
            model.matvec_slice(tensor, input.data(), actual.data(), 13, 129);
            require(std::memcmp(actual.data(), expected.data() + 13, 129 * sizeof(float)) == 0,
                    "slice scheduling changed dot products");
        }
        std::puts("PASS DS41 parallel coverage, nested dispatch, bitwise F32/Q8 rows and slices");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
