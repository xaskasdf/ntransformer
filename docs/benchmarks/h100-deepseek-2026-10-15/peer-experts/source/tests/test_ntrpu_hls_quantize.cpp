#include "ntrpu_quantize_q8_0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::quantize_hls;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

struct Fixture {
    std::array<float, MAX_COLUMNS> input{};
    std::array<int8_t, MAX_COLUMNS> output{};
    std::array<double, MAX_BLOCKS> scales{};
    Fixture() { output.fill(-128); scales.fill(-9876.5); }
    uint32_t run(uint32_t columns) {
        return ntrpu_quantize_q8_0(input.data(), output.data(), scales.data(), columns);
    }
    void check(uint32_t columns) {
        const auto saved = input;
        const auto old_output = output;
        const auto old_scales = scales;
        CHECK(run(columns) == OK);
        CHECK(std::memcmp(input.data(), saved.data(), sizeof(input)) == 0);
        for (uint32_t block = 0; block < columns / BLOCK_ELEMENTS; ++block) {
            double maximum = 0;
            for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
                maximum = std::max(maximum, std::abs(double(input[block * BLOCK_ELEMENTS + lane])));
            }
            const double expected_scale = maximum / 127.0;
            CHECK(scales[block] == expected_scale);
            for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
                const uint32_t column = block * BLOCK_ELEMENTS + lane;
                const double value = expected_scale == 0 ? 0 : double(input[column]) / expected_scale;
                const double expected = std::max(-127.0, std::min(127.0, std::round(value)));
                CHECK(output[column] == expected);
                CHECK(output[column] != -128);
                CHECK(std::abs(double(input[column]) - scales[block] * output[column]) <=
                    scales[block] * 0.5 + maximum * 1e-15);
            }
        }
        for (uint32_t column = columns; column < MAX_COLUMNS; ++column) CHECK(output[column] == old_output[column]);
        for (uint32_t block = columns / BLOCK_ELEMENTS; block < MAX_BLOCKS; ++block) CHECK(scales[block] == old_scales[block]);
    }
};

static void shapes() {
    Fixture f;
    for (uint32_t c = 0; c < MAX_COLUMNS; ++c) f.input[c] = float(int(c % 257) - 128) * float(1 + c / 32);
    for (uint32_t columns : {32U, 64U, 256U, 736U, 1024U, 32U}) f.check(columns);
    f.input.fill(-0.0f);
    f.check(MAX_COLUMNS);
    for (auto value : f.output) CHECK(value == 0);
    for (auto scale : f.scales) CHECK(scale == 0 && !std::signbit(scale));
}

static void ties() {
    Fixture f;
    f.input[31] = 127;
    for (int integer = 0; integer < 127; ++integer) {
        const float tie = float(integer) + 0.5f;
        const float below = std::nextafter(tie, 0.0f);
        const float above = std::nextafter(tie, std::numeric_limits<float>::infinity());
        f.input[0] = tie; f.input[1] = below; f.input[2] = above;
        f.input[3] = -tie; f.input[4] = -below; f.input[5] = -above;
        f.check(32);
        CHECK(f.scales[0] == 1);
        CHECK(f.output[0] == integer + 1 && f.output[1] == integer && f.output[2] == integer + 1);
        CHECK(f.output[3] == -integer - 1 && f.output[4] == -integer && f.output[5] == -integer - 1);
    }
}

static void dynamic() {
    Fixture f;
    uint32_t seed = 41287;
    for (uint32_t trial = 0; trial < 16; ++trial) {
        for (float& value : f.input) {
            seed = seed * 1664525U + 1013904223U;
            uint32_t bits = seed;
            if ((bits & 0x7f800000U) == 0x7f800000U) bits ^= 0x00800000U;
            std::memcpy(&value, &bits, sizeof(value));
        }
        f.check(MAX_COLUMNS);
    }
    for (float maximum : {std::numeric_limits<float>::denorm_min(),
                          std::nextafter(std::numeric_limits<float>::min(), 0.0f),
                          std::numeric_limits<float>::min(), std::numeric_limits<float>::max()}) {
        f.input.fill(maximum);
        f.input[0] = -maximum;
        f.input[1] = 0;
        f.check(32);
        CHECK(f.output[0] == -127 && f.output[1] == 0 && f.output[2] == 127);
        CHECK(f.scales[0] > 0 && std::isfinite(f.scales[0]));
    }
}

static void invalid(bool host_only) {
    Fixture f;
    const auto output = f.output;
    const auto scales = f.scales;
    for (uint32_t columns : {0U, 1U, 31U, 33U, 1025U, UINT32_MAX}) {
        CHECK(f.run(columns) == INVALID_SHAPE);
        CHECK(f.output == output && f.scales == scales);
        // Only CPU-specific invocation tests null pointers; default RTL bench uses full buffers.
        if (host_only) CHECK(ntrpu_quantize_q8_0(nullptr, nullptr, nullptr, columns) == INVALID_SHAPE);
    }
    for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
        f.input[MAX_COLUMNS - 1] = bad;
        CHECK(f.run(MAX_COLUMNS) == NONFINITE_INPUT);
        CHECK(f.output == output && f.scales == scales);
    }
    // Invalid data outside the requested range must not be accessed.
    f.check(32);
    f.input[MAX_COLUMNS - 1] = 1;
    f.check(MAX_COLUMNS);
}

static void subnormals() {
    Fixture f;
    // One block for each leading subnormal bit, then exponent boundaries.
    // Vary signs and mantissas within each block; keep the oracle as native FP64.
    for (uint32_t block = 0; block < MAX_BLOCKS; ++block) {
        const uint32_t leading = std::min(block, 22U);
        const uint32_t peak = block < 23 ? (1U << leading) :
            (block == 23 ? 0x007fffffU : (block == 24 ? 0x00800000U : 0x00800001U));
        for (uint32_t lane = 0; lane < BLOCK_ELEMENTS; ++lane) {
            uint32_t bits = lane < 2 ? peak : uint32_t(uint64_t(peak) * (lane - 2) / 29U);
            if (lane & 1U) bits |= 0x80000000U;
            std::memcpy(&f.input[block * BLOCK_ELEMENTS + lane], &bits, sizeof(bits));
        }
    }
    f.check(MAX_COLUMNS);
    // Same buffers, opposite signs; exercise repeated invocation and signed zero.
    for (float& value : f.input) value = -value;
    f.check(MAX_COLUMNS);
}

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if (which == "all" || which == "shapes") shapes();
        if (which == "all" || which == "ties") ties();
        if (which == "all" || which == "dynamic") dynamic();
        if (which == "all" || which == "subnormals") subnormals();
        if (which == "all" || which == "invalid") invalid(which == "invalid");
        if (which != "all" && which != "shapes" && which != "ties" && which != "dynamic" &&
            which != "subnormals" && which != "invalid") {
            throw std::runtime_error("unknown test");
        }
        std::cout << "PASS Q8 activation quantization " << which << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
