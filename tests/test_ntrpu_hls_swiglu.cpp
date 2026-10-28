#include "ntrpu_swiglu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::swiglu_hls;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

static uint32_t bits(float value) {
    uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}

static float from_bits(uint32_t value) {
    float result; std::memcpy(&result, &value, sizeof(result)); return result;
}

static void numeric() {
    std::array<float, MAX_ELEMENTS> gate{}, up{}, output{};
    for (uint32_t i = 0; i < MAX_ELEMENTS; ++i) {
        gate[i] = float(int(i) - 512) / 16.0f;
        up[i] = float(int(i % 37) - 18) / 7.0f;
    }
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float large = std::numeric_limits<float>::max();
    const std::array<float, 16> edge_gate{{0.0f, -0.0f, tiny, -tiny, from_bits(0x007fffff),
        from_bits(0x00800000), -128.0f, -256.0f, -257.0f, large, -large, 1.0f, -1.0f, 256, 257, -80}};
    const std::array<float, 16> edge_up{{1, -1, large, large, 1, 1, large, large, large,
        1, 1, tiny, -tiny, 1, 1, large}};
    std::copy(edge_gate.begin(), edge_gate.end(), gate.begin());
    std::copy(edge_up.begin(), edge_up.end(), up.begin());
    for (uint32_t count : {MAX_ELEMENTS, 1U, 33U}) {
        output.fill(-8765.0f);
        CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), count) == OK);
        for (uint32_t i = 0; i < count; ++i) {
            const double g = gate[i], u = up[i], e = std::exp(-std::abs(g));
            const float expected = float((g * (g >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e))) * u);
            CHECK(std::isfinite(output[i]));
            const double tolerance = std::max(2.0 * double(tiny), 2e-7 * std::abs(double(expected)));
            CHECK(std::abs(double(output[i]) - expected) <= tolerance);
            if (expected == 0) CHECK(bits(output[i]) == bits(expected));
        }
        for (uint32_t i = count; i < MAX_ELEMENTS; ++i) CHECK(output[i] == -8765.0f);
    }
}

static void invalid() {
    std::array<float, MAX_ELEMENTS> gate{}, up{}, output{};
    gate.fill(1); up.fill(1); output.fill(-8765);
    const auto original = output;
    for (uint32_t count : {0U, MAX_ELEMENTS + 1}) {
        CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), count) == INVALID_SHAPE);
        CHECK(output == original);
    }
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
        gate[32] = value;
        CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), 33) == NONFINITE_INPUT);
        CHECK(output == original);
        gate[32] = 1; up[32] = value;
        CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), 33) == NONFINITE_INPUT);
        CHECK(output == original); up[32] = 1;
    }
    gate[32] = up[32] = std::numeric_limits<float>::max();
    CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), 33) == OUTPUT_OVERFLOW);
    CHECK(output == original);
    // Input validation has priority over an earlier overflow, regardless of order.
    gate[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), 33) == NONFINITE_INPUT);
    CHECK(output == original);
    gate.fill(0); up.fill(1);
    CHECK(ntrpu_swiglu(gate.data(), up.data(), output.data(), 33) == OK);
    for (uint32_t i = 0; i < 33; ++i) CHECK(output[i] == 0);
}

static void conversions() {
    // Audit every signed FP32 subnormal and zero, not just representative values.
    for (uint32_t sign : {0U, 0x80000000U}) {
        for (uint32_t fraction = 0; fraction < 0x00800000U; ++fraction) {
            const uint32_t pattern = sign | fraction;
            const float input = from_bits(pattern);
            double wide = 0;
            float narrow = 1;
            CHECK(detail::widen(input, wide));
            const double expected = double(input);
            CHECK(std::memcmp(&wide, &expected, sizeof(wide)) == 0);
            CHECK(detail::narrow(wide, narrow));
            CHECK(bits(narrow) == pattern);
        }
    }
    auto check_rounding = [](double value) {
        float result = 123;
        const bool valid = std::isfinite(value) && std::abs(value) <= double(std::numeric_limits<float>::max());
        CHECK(detail::narrow(value, result) == valid);
        if (valid) CHECK(bits(result) == bits(float(value)));
        else CHECK(result == 123);
    };
    uint32_t state = 20261001;
    for (uint32_t i = 0; i < 100000; ++i) {
        state = state * 1664525U + 1013904223U;
        const uint32_t pattern = (state & 0x7fffffffU) % 0x7f7fffffU;
        const double midpoint = (double(from_bits(pattern)) + double(from_bits(pattern + 1))) * 0.5;
        for (double sign : {1.0, -1.0}) {
            check_rounding(sign * midpoint);
            check_rounding(sign * std::nextafter(midpoint, 0.0));
            check_rounding(sign * std::nextafter(midpoint, std::numeric_limits<double>::infinity()));
        }
    }
    for (double value : {0.0, -0.0, std::numeric_limits<double>::denorm_min(),
        double(std::numeric_limits<float>::max()), std::nextafter(double(std::numeric_limits<float>::max()),
        std::numeric_limits<double>::infinity()), std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) check_rounding(value);
}

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if (which == "all" || which == "numeric") numeric();
        if (which == "all" || which == "invalid") invalid();
        // The exhaustive host audit is deliberately separate from the RTL bench.
        if (which == "conversions") conversions();
        CHECK(which == "all" || which == "numeric" || which == "invalid" || which == "conversions");
        std::cout << "PASS SwiGLU " << which << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
