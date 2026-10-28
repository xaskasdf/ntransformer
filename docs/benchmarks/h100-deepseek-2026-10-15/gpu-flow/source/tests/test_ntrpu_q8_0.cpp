#include "ntrpu/q8_0.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ntrpu;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

template<class Function>
static void invalid(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}

static void append_block(std::vector<std::byte>& bytes, uint16_t scale,
                         const std::vector<int8_t>& values) {
    CHECK(values.size() == 32);
    bytes.push_back(static_cast<std::byte>(scale & 255U));
    bytes.push_back(static_cast<std::byte>(scale >> 8));
    for (auto value : values) bytes.push_back(static_cast<std::byte>(static_cast<uint8_t>(value)));
}

static void test_scales() {
    std::vector<std::byte> weights;
    append_block(weights, 0x3800, std::vector<int8_t>(32, 1)); // 0.5
    append_block(weights, 0x4000, std::vector<int8_t>(32, 1)); // 2.0
    std::vector<float> input(64, 127);
    std::fill(input.begin() + 32, input.end(), -127.0f);
    CHECK(q8_0_gemv_reference(weights, input, 1)[0] == -6096);
    CHECK(q8_0_gemv_hls_model(weights, input, 1)[0] == -6096);
    // Combining the two INT32 dots before scaling would incorrectly give zero.
    const uint16_t bits[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff};
    const double expected[] = {0, 0, std::ldexp(1.0, -24), std::ldexp(1023.0, -24),
                               std::ldexp(1.0, -14), 1, -1, 65504};
    for (size_t i = 0; i < 8; ++i) {
        weights.clear();
        auto q = std::vector<int8_t>(32, 0);
        q[0] = -128;
        append_block(weights, bits[i], q);
        input.assign(32, 0);
        input[0] = 127;
        CHECK(q8_0_gemv_reference(weights, input, 1)[0] == expected[i] * -128 * 127);
        CHECK(q8_0_gemv_hls_model(weights, input, 1)[0] == expected[i] * -128 * 127);
    }
}

static void test_quantization() {
    std::vector<std::byte> weights;
    std::vector<int8_t> q(32, 0);
    q[0] = 1;
    q[1] = 2;
    append_block(weights, 0x3c00, q);
    std::vector<float> input(32, 0);
    input[0] = 0.5f;
    input[1] = -0.5f;
    input[2] = 127; // Makes the activation scale exactly one.
    CHECK(q8_0_gemv_reference(weights, input, 1)[0] == -0.5);
    CHECK(q8_0_gemv_hls_model(weights, input, 1)[0] == -1); // Ties away from zero.
    input.assign(32, 0);
    CHECK(q8_0_gemv_hls_model(weights, input, 1)[0] == 0);
    input[0] = std::numeric_limits<float>::denorm_min();
    const auto tiny = q8_0_gemv_hls_model(weights, input, 1)[0];
    CHECK(std::abs(tiny - double(input[0])) <= double(input[0]) * 1e-12);
    input[0] = std::numeric_limits<float>::max();
    const auto large = q8_0_gemv_hls_model(weights, input, 1)[0];
    CHECK(std::isfinite(large));
    CHECK(std::abs(large - double(input[0])) <= double(input[0]) * 1e-12);
}

static void test_quality() {
    constexpr size_t rows = 35, blocks = 3, columns = blocks * 32;
    std::mt19937 random(20260930);
    std::vector<float> input(columns);
    for (auto& value : input) value = (static_cast<int>(random() % 20001) - 10000) / 1000.0f;
    std::vector<std::byte> weights;
    std::vector<double> golden(rows, 0), error_bound(rows, 0);
    const uint16_t bits[] = {0x3400, 0x3800, 0x4000};
    const double scales[] = {0.25, 0.5, 2};
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            std::vector<int8_t> q(32);
            double maximum = 0;
            for (size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(double(input[block * 32 + j])));
            for (size_t j = 0; j < 32; ++j) {
                q[j] = static_cast<int8_t>(static_cast<int>(random() % 256) - 128);
                const double w = scales[block] * q[j];
                golden[row] += w * double(input[block * 32 + j]);
                error_bound[row] += std::abs(w) * maximum / 254.0;
            }
            append_block(weights, bits[block], q);
        }
    }
    const auto reference = q8_0_gemv_reference(weights, input, rows);
    const auto computed = q8_0_gemv_hls_model(weights, input, rows);
    double max_error = 0, squared_error = 0, squared_reference = 0;
    for (size_t row = 0; row < rows; ++row) {
        CHECK(std::abs(reference[row] - golden[row]) < 1e-10);
        const double error = std::abs(computed[row] - golden[row]);
        CHECK(error <= error_bound[row] + 1e-8);
        max_error = std::max(max_error, error);
        squared_error += error * error;
        squared_reference += golden[row] * golden[row];
    }
    CHECK(max_error > 0); // This quantized path is explicitly approximate.
    std::cout << "Synthetic Q8_0 35x96: max absolute error=" << max_error
              << ", relative L2 error=" << std::sqrt(squared_error / squared_reference) << '\n';
}

static void test_validation() {
    std::vector<std::byte> weights;
    append_block(weights, 0x3c00, std::vector<int8_t>(32, 1));
    std::vector<float> input(32, 1);
    const auto check_both = [&](std::span<const std::byte> w, std::span<const float> x, size_t rows) {
        invalid([&] { q8_0_gemv_reference(w, x, rows); });
        invalid([&] { q8_0_gemv_hls_model(w, x, rows); });
    };
    check_both(weights, input, 0);
    check_both(weights, input, std::numeric_limits<size_t>::max());
    check_both(std::span<const std::byte>(weights).first(33), input, 1);
    check_both(weights, std::span<const float>(input).first(31), 1);
    check_both(weights, {}, 1);
    for (float value : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        input[0] = value;
        check_both(weights, input, 1);
    }
    input[0] = 1;
    for (uint16_t scale : {uint16_t{0x7c00}, uint16_t{0xfc00}, uint16_t{0x7e00}}) {
        weights[0] = static_cast<std::byte>(scale & 255U);
        weights[1] = static_cast<std::byte>(scale >> 8);
        check_both(weights, input, 1);
    }
}

int main(int argc, char** argv) {
    struct Case { const char* name; void (*run)(); };
    const Case cases[] = {{"scales", test_scales}, {"quantization", test_quantization},
                          {"quality", test_quality}, {"validation", test_validation}};
    size_t ran = 0, failed = 0;
    if (argc > 2) return 2;
    for (const auto& test : cases) {
        if (argc == 2 && std::string(argv[1]) != test.name) continue;
        ++ran;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    return ran != 0 && failed == 0 ? 0 : 1;
}
