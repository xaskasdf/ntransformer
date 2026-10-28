#include "ntrpu_gemv_fp32_q8_0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::fp32_q8_hls;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

struct Fixture {
    std::array<uint8_t, 34816> weights{};
    std::array<float, 1024> input{};
    std::array<double, 32> output{};
    Fixture() { output.fill(-9876.5); }
    uint32_t run(uint32_t rows, uint32_t columns) {
        return ntrpu_gemv_fp32_q8_0(weights.data(), input.data(), output.data(), rows, columns);
    }
    void check(uint32_t rows, uint32_t columns) {
        std::array<int32_t, 1024> quantized{};
        std::array<double, 32> scales{};
        for (uint32_t b = 0; b < columns / 32; ++b) {
            double peak = 0;
            for (uint32_t lane = 0; lane < 32; ++lane) peak = std::max(peak, std::abs(double(input[b * 32 + lane])));
            scales[b] = peak / 127.0;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                quantized[b * 32 + lane] = scales[b] == 0 ? 0 : int32_t(std::max(-127.0,
                    std::min(127.0, std::round(double(input[b * 32 + lane]) / scales[b]))));
            }
        }
        const auto old = output;
        CHECK(run(rows, columns) == OK);
        for (uint32_t row = 0; row < rows; ++row) {
            double expected = 0;
            for (uint32_t b = 0; b < columns / 32; ++b) {
                const auto* w = weights.data() + (row * (columns / 32) + b) * 34;
                const uint16_t half = uint16_t(w[0]) | (uint16_t(w[1]) << 8);
                const uint32_t exponent = (half >> 10) & 31U;
                const uint32_t fraction = half & 1023U;
                CHECK(exponent != 31);
                double weight_scale = exponent == 0 ? std::ldexp(double(fraction), -24) :
                    std::ldexp(double(1024U + fraction), int(exponent) - 25);
                if (half & 0x8000U) weight_scale = -weight_scale;
                int64_t dot = 0;
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const int value = w[2 + lane] < 128 ? int(w[2 + lane]) : int(w[2 + lane]) - 256;
                    dot += int64_t(value) * quantized[b * 32 + lane];
                }
                expected += (weight_scale * scales[b]) * double(dot);
            }
            CHECK(std::isfinite(output[row]));
            CHECK(output[row] == expected);
        }
        for (uint32_t row = rows; row < 32; ++row) CHECK(output[row] == old[row]);
    }
    void fill(uint32_t rows, uint32_t columns) {
        for (uint32_t c = 0; c < columns; ++c) input[c] = float(int(c % 255) - 127) / 8.0f;
        for (uint32_t b = 0; b < rows * (columns / 32); ++b) {
            const uint16_t half = uint16_t((b & 1U ? 0x8000U : 0) | (b % 3 == 0 ? 1U : 0x3800U));
            weights[b * 34] = uint8_t(half); weights[b * 34 + 1] = uint8_t(half >> 8);
            for (uint32_t lane = 0; lane < 32; ++lane) weights[b * 34 + 2 + lane] = uint8_t((b * 29 + lane * 7) & 255);
        }
    }
};

static void numeric() {
    Fixture f;
    f.fill(32, 1024); f.check(32, 1024);
    f.fill(3, 736); f.check(3, 736);
    // Exercise input subnormal widening through the connected consumer as well.
    f.fill(2, 32);
    for (uint32_t c = 0; c < 32; ++c) {
        uint32_t bits = (1U << (c % 23)) | (c & 1U ? 0x80000000U : 0);
        std::memcpy(&f.input[c], &bits, sizeof(bits));
    }
    f.check(2, 32);
    f.input.fill(-0.0f); f.check(2, 32);
    f.fill(1, 32); f.check(1, 32);
}

static void invalid() {
    Fixture f;
    const auto untouched = f.output;
    for (uint32_t rows : {0U, 33U}) { CHECK(f.run(rows, 32) == INVALID_SHAPE); CHECK(f.output == untouched); }
    for (uint32_t columns : {0U, 31U, 1056U}) { CHECK(f.run(1, columns) == INVALID_SHAPE); CHECK(f.output == untouched); }
    f.fill(2, 64);
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        f.input[63] = bad;
        CHECK(f.run(2, 64) == NONFINITE_INPUT); CHECK(f.output == untouched);
    }
    f.fill(2, 64);
    // Last weight scale fails after an earlier row has already been computed.
    f.weights[3 * 34] = 0; f.weights[3 * 34 + 1] = 0x7c;
    CHECK(f.run(2, 64) == INVALID_WEIGHT_SCALE); CHECK(f.output == untouched);
    f.fill(2, 64); f.check(2, 64);
}

static uint32_t read_u32(std::istream& file) {
    unsigned char bytes[4];
    CHECK(bool(file.read(reinterpret_cast<char*>(bytes), 4)));
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) | (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

static void brandon(const char* path) {
    std::ifstream file(path, std::ios::binary);
    char magic[8]; CHECK(bool(file.read(magic, 8))); CHECK(std::memcmp(magic, "NTRQ8T01", 8) == 0);
    const uint32_t count = read_u32(file); CHECK(count > 0 && count <= 16);
    for (uint32_t tile = 0; tile < count; ++tile) {
        Fixture f;
        const uint32_t rows = read_u32(file), columns = read_u32(file);
        CHECK(rows > 0 && rows <= 32 && columns > 0 && columns <= 1024 && columns % 32 == 0);
        CHECK(bool(file.read(reinterpret_cast<char*>(f.weights.data()), rows * (columns / 32) * 34)));
        for (uint32_t c = 0; c < columns; ++c) {
            const uint32_t bits = read_u32(file); std::memcpy(&f.input[c], &bits, sizeof(bits));
            CHECK(std::isfinite(f.input[c]));
        }
        f.check(rows, columns);
    }
    CHECK(file.peek() == std::char_traits<char>::eof());
    std::cout << "PASS Brandon connected tiles " << count << '\n';
}

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if (which == "all" || which == "--fixture" || which == "numeric") numeric();
        if (which == "all" || which == "--fixture" || which == "invalid") invalid();
        if (which == "--fixture") { CHECK(argc == 3); brandon(argv[2]); }
        CHECK(which == "all" || which == "--fixture" || which == "numeric" || which == "invalid");
        std::cout << "PASS connected FP32/Q8 GEMV " << which << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
