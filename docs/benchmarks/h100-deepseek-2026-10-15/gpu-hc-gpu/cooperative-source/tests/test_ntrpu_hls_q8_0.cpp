// Shared CPU / Vitis C simulation / C-RTL testbench. No external test framework.
#include "ntrpu_gemv_q8_0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::q8_hls;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

struct Fixture {
    std::array<uint8_t, MAX_WEIGHT_BYTES> weights{};
    std::array<int8_t, MAX_COLUMNS> input{};
    std::array<double, MAX_BLOCKS> scales{};
    std::array<double, MAX_ROWS> output{};
    Fixture() { output.fill(1234567.25); }
    void scale(uint32_t block, uint16_t bits) {
        weights[block * BLOCK_BYTES] = uint8_t(bits & 255);
        weights[block * BLOCK_BYTES + 1] = uint8_t(bits >> 8);
    }
    uint32_t run(uint32_t rows, uint32_t columns) {
        return ntrpu_gemv_q8_0(weights.data(), input.data(), scales.data(), output.data(), rows, columns);
    }
};

static double reference_scale(uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    const double value = exponent == 0 ? std::ldexp(double(fraction), -24) :
                                        std::ldexp(double(1024 + fraction), exponent - 25);
    return bits & 0x8000 ? -value : value;
}

static void numeric() {
    uint32_t seed = 4139;
    auto next = [&seed]() { seed = seed * 1664525U + 1013904223U; return seed >> 16; };
    for (uint32_t rows : {1U, 3U, 32U, 2U}) {
        for (uint32_t columns : {32U, 64U, 736U, 1024U}) {
            Fixture f;
            const uint32_t blocks = columns / BLOCK_ELEMENTS;
            for (uint32_t c = 0; c < columns; ++c) f.input[c] = int8_t(int(next() % 256) - 128);
            for (uint32_t b = 0; b < blocks; ++b) f.scales[b] = (b % 5 == 0) ? 0 : (next() % 1000) / 127.0;
            std::array<double, MAX_ROWS> expected{};
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t b = 0; b < blocks; ++b) {
                    const uint16_t bits = uint16_t((next() % 0x7c00) | ((b & 1) << 15));
                    const uint32_t index = r * blocks + b;
                    f.scale(index, bits);
                    int64_t dot = 0;
                    for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
                        const int w = int(next() % 256) - 128;
                        f.weights[index * BLOCK_BYTES + 2 + lane] = uint8_t(w);
                        dot += int64_t(w) * f.input[b * BLOCK_ELEMENTS + lane];
                    }
                    expected[r] += reference_scale(bits) * f.scales[b] * double(dot);
                }
            }
            const auto weights = f.weights;
            const auto input = f.input;
            const auto scales = f.scales;
            CHECK(f.run(rows, columns) == OK);
            CHECK(f.weights == weights && f.input == input && f.scales == scales);
            for (uint32_t r = 0; r < rows; ++r) {
                CHECK(std::abs(f.output[r] - expected[r]) <= 1e-12 * std::max(1.0, std::abs(expected[r])));
            }
            for (uint32_t r = rows; r < MAX_ROWS; ++r) CHECK(f.output[r] == 1234567.25);
        }
    }
    Fixture f;
    f.scales.fill(1);
    f.input.fill(127);
    f.scale(0, 0x3800); // 0.5
    f.scale(1, 0x4000); // 2.0
    for (uint32_t j = 0; j < 32; ++j) {
        f.weights[2 + j] = 1;
        f.weights[36 + j] = 1;
        f.input[32 + j] = -127;
    }
    CHECK(f.run(1, 64) == OK);
    CHECK(f.output[0] == -6096); // A single scale applied after both blocks is incorrect.
    f.weights.fill(0x80);
    f.input.fill(-128);
    f.scale(0, 0x3c00);
    CHECK(f.run(1, 32) == OK);
    CHECK(f.output[0] == 524288);
}

static void check_scale(uint16_t bits) {
    Fixture f;
    f.scale(0, bits);
    f.weights[2] = 0x80;
    f.input[0] = 127;
    f.scales[0] = 1;
    if ((bits & 0x7c00) == 0x7c00) {
        CHECK(f.run(1, 32) == INVALID_SCALE);
        CHECK(f.output[0] == 1234567.25);
    } else {
        CHECK(f.run(1, 32) == OK);
        CHECK(f.output[0] == reference_scale(bits) * -128 * 127);
    }
}

static void scales() {
    for (uint32_t exponent = 0; exponent < 32; ++exponent) {
        check_scale(uint16_t((exponent << 10) | 1));
        check_scale(uint16_t(0x8000 | (exponent << 10) | 1023));
    }
    check_scale(0);
    check_scale(0x8000);
    Fixture f;
    f.scale(0, 0x7bff);
    f.weights[2] = 127;
    f.input[0] = 127;
    for (double value : {double(std::numeric_limits<float>::denorm_min()) / 127.0,
                         double(std::numeric_limits<float>::max()) / 127.0}) {
        f.scales[0] = value;
        CHECK(f.run(1, 32) == OK);
        CHECK(std::isfinite(f.output[0]));
        CHECK(f.output[0] == (65504.0 * value) * (127 * 127));
    }
}

static void invalid(bool host_only) {
    Fixture f;
    const auto untouched = f.output;
    for (uint32_t rows : {0U, 33U, UINT32_MAX}) {
        CHECK(f.run(rows, 32) == INVALID_SHAPE);
        CHECK(f.output == untouched);
        // RTL harnesses marshal full-depth buffers even for rejected commands.
        if (host_only) CHECK(ntrpu_gemv_q8_0(nullptr, nullptr, nullptr, nullptr, rows, 32) == INVALID_SHAPE);
    }
    for (uint32_t columns : {0U, 1U, 31U, 33U, 1025U, UINT32_MAX}) {
        CHECK(f.run(1, columns) == INVALID_SHAPE);
        CHECK(f.output == untouched);
        if (host_only) CHECK(ntrpu_gemv_q8_0(nullptr, nullptr, nullptr, nullptr, 1, columns) == INVALID_SHAPE);
    }
    f.scales.fill(1);
    // Invalid scale at the very end must not commit preceding valid rows.
    f.scale(MAX_ROWS * MAX_BLOCKS - 1, 0x7c00);
    CHECK(f.run(MAX_ROWS, MAX_COLUMNS) == INVALID_SCALE);
    CHECK(f.output == untouched);
    f.scale(MAX_ROWS * MAX_BLOCKS - 1, 0x3c00);
    for (double bad : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max()}) {
        f.scales[MAX_BLOCKS - 1] = bad;
        CHECK(f.run(MAX_ROWS, MAX_COLUMNS) == INVALID_SCALE);
        CHECK(f.output == untouched);
    }
    f.scales.fill(0);
    CHECK(f.run(MAX_ROWS, MAX_COLUMNS) == OK);
    for (double value : f.output) CHECK(value == 0);
}

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if (which == "all" || which == "numeric") numeric();
        if (which == "all" || which == "scales") scales();
        if (which == "all" || which == "invalid") invalid(which == "invalid");
        // Host-only exhaustive decoder validation, not thousands of RTL transactions.
        if (which == "exhaustive") for (uint32_t bits = 0; bits < 65536; ++bits) check_scale(uint16_t(bits));
        if (which != "all" && which != "numeric" && which != "scales" &&
            which != "invalid" && which != "exhaustive") throw std::runtime_error("unknown test");
        std::cout << "PASS Q8_0 HLS " << which << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
