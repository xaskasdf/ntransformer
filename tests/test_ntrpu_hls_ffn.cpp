#include "ntrpu_ffn_q8_0.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ntrpu::ffn_hls;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)

// Independent oracle: no HLS quantizer, GEMV, activation or conversion helpers.
static std::vector<float> reference_gemv(const std::vector<uint8_t>& weights,
    const std::vector<float>& input, uint32_t rows, uint32_t columns) {
    std::vector<int32_t> q(columns);
    std::vector<double> scales(columns / 32);
    for (uint32_t b = 0; b < columns / 32; ++b) {
        double maximum = 0;
        for (uint32_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(double(input[b * 32 + j])));
        scales[b] = maximum / 127.0;
        for (uint32_t j = 0; j < 32; ++j) q[b * 32 + j] = maximum == 0 ? 0 :
            int32_t(std::max(-127.0, std::min(127.0, std::round(double(input[b * 32 + j]) / scales[b]))));
    }
    std::vector<float> output(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        double sum = 0;
        for (uint32_t b = 0; b < columns / 32; ++b) {
            const uint8_t* w = weights.data() + (r * (columns / 32) + b) * 34;
            const uint16_t half = uint16_t(w[0]) | (uint16_t(w[1]) << 8);
            const uint32_t exponent = (half >> 10) & 31U, fraction = half & 1023U;
            CHECK(exponent != 31);
            double scale = exponent == 0 ? std::ldexp(double(fraction), -24) :
                std::ldexp(double(1024U + fraction), int(exponent) - 25);
            if (half & 0x8000U) scale = -scale;
            int64_t dot = 0;
            for (uint32_t j = 0; j < 32; ++j) dot +=
                int64_t(w[j + 2] < 128 ? int(w[j + 2]) : int(w[j + 2]) - 256) * q[b * 32 + j];
            sum += (scale * scales[b]) * double(dot);
        }
        CHECK(std::isfinite(sum) && std::abs(sum) <= double(std::numeric_limits<float>::max()));
        output[r] = float(sum);
    }
    return output;
}

struct Fixture {
    std::vector<uint8_t> gate = std::vector<uint8_t>(MAX_GATE_BYTES);
    std::vector<uint8_t> up = std::vector<uint8_t>(MAX_GATE_BYTES);
    std::vector<uint8_t> down = std::vector<uint8_t>(MAX_DOWN_BYTES);
    std::vector<float> input = std::vector<float>(MAX_DIM);
    std::vector<float> output = std::vector<float>(MAX_DIM, -9876.5f);
    uint32_t run(uint32_t dim, uint32_t hidden) {
        return ntrpu_ffn_q8_0(gate.data(), up.data(), down.data(), input.data(), output.data(), dim, hidden);
    }
    void fill(uint32_t dim, uint32_t hidden) {
        const uint32_t padded = (hidden + 31) / 32 * 32;
        uint32_t seed = 17;
        auto matrix = [&](std::vector<uint8_t>& w, uint32_t count) {
            for (uint32_t b = 0; b < count; ++b) {
                const uint16_t half = uint16_t((b % 2 ? 0x8000U : 0) | 0x2000U);
                w[b * 34] = uint8_t(half); w[b * 34 + 1] = uint8_t(half >> 8);
                for (uint32_t j = 0; j < 32; ++j) {
                    seed = seed * 1664525U + 1013904223U;
                    w[b * 34 + j + 2] = uint8_t(seed >> 24);
                }
            }
        };
        matrix(gate, hidden * (dim / 32)); matrix(up, hidden * (dim / 32));
        // Nonzero down weights in padding columns expose stale/uninitialized padding.
        matrix(down, dim * (padded / 32));
        for (uint32_t i = 0; i < dim; ++i) input[i] = float(int(i % 29) - 14) / 19.0f;
        std::fill(output.begin(), output.end(), -9876.5f);
    }
    void check(uint32_t dim, uint32_t hidden) {
        auto g = reference_gemv(gate, input, hidden, dim);
        const auto u = reference_gemv(up, input, hidden, dim);
        for (uint32_t i = 0; i < hidden; ++i) {
            const double value = g[i], e = std::exp(-std::abs(value));
            g[i] = float((value * (value >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e))) * double(u[i]));
        }
        g.resize((hidden + 31) / 32 * 32, 0.0f);
        const auto expected = reference_gemv(down, g, dim, uint32_t(g.size()));
        const auto old = output;
        CHECK(run(dim, hidden) == OK);
        for (uint32_t i = 0; i < dim; ++i) {
            CHECK(std::isfinite(output[i]));
            CHECK(std::abs(double(output[i]) - expected[i]) <= 1e-6 + 1e-5 * std::abs(double(expected[i])));
        }
        for (size_t i = dim; i < old.size(); ++i) CHECK(output[i] == old[i]);
    }
};

static void numeric(bool include_maximum = true) {
    Fixture f;
    if (include_maximum) { f.fill(256, 720); f.check(256, 720); }
    f.fill(64, 65); f.check(64, 65);
    f.fill(32, 33); f.check(32, 33);
    f.fill(32, 2); f.check(32, 2);
    std::fill(f.input.begin(), f.input.end(), -0.0f); f.check(32, 2);
    for (uint32_t i = 0; i < 32; ++i) f.input[i] = float(int(i) - 16) * std::numeric_limits<float>::denorm_min();
    f.check(32, 2);
    // Repeat a padded shape after larger and smaller commands.
    f.fill(32, 33); f.check(32, 33);
}

static void invalid() {
    Fixture f;
    const auto untouched = f.output;
    for (uint32_t dim : {0U, 31U, MAX_DIM + 32}) {
        CHECK(f.run(dim, 2) == INVALID_SHAPE); CHECK(f.output == untouched);
    }
    for (uint32_t hidden : {0U, 1U, MAX_HIDDEN + 1}) {
        CHECK(f.run(32, hidden) == INVALID_SHAPE); CHECK(f.output == untouched);
    }
    f.fill(64, 33);
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        f.input[63] = bad;
        CHECK(f.run(64, 33) == NONFINITE_INPUT); CHECK(f.output == untouched);
    }
    for (unsigned stage = 0; stage < 3; ++stage) {
        f.fill(64, 33);
        auto& w = stage == 0 ? f.gate : stage == 1 ? f.up : f.down;
        const uint32_t blocks = stage < 2 ? 33 * 2 : 64 * 2;
        w[(blocks - 1) * 34] = 0; w[(blocks - 1) * 34 + 1] = 0x7c;
        CHECK(f.run(64, 33) == INVALID_WEIGHT_SCALE); CHECK(f.output == untouched);
    }
    f.fill(32, 2);
    std::fill(f.input.begin(), f.input.end(), std::numeric_limits<float>::max());
    for (uint32_t b = 0; b < 2; ++b) {
        f.gate[b * 34] = 0xff; f.gate[b * 34 + 1] = 0x7b;
        std::fill(f.gate.begin() + b * 34 + 2, f.gate.begin() + (b + 1) * 34, uint8_t(127));
    }
    CHECK(f.run(32, 2) == NUMERIC_OVERFLOW); CHECK(f.output == untouched);
    auto sparse_projection = [](std::vector<uint8_t>& weights, uint32_t rows, uint32_t columns) {
        std::fill(weights.begin(), weights.end(), uint8_t(0));
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t offset = r * (columns / 32) * 34;
            weights[offset + 1] = 0x3c; // scale 1, first coefficient 1
            weights[offset + 2] = 1;
        }
    };
    // Finite gate/up results can still overflow at SwiGLU multiplication.
    f.fill(64, 33);
    sparse_projection(f.gate, 33, 64); sparse_projection(f.up, 33, 64);
    std::fill(f.input.begin(), f.input.end(), 0.0f); f.input[0] = 1e20f;
    CHECK(f.run(64, 33) == NUMERIC_OVERFLOW); CHECK(f.output == untouched);
    // Earlier down tiles succeed; only the final row overflows during narrowing.
    f.input[0] = 1e17f;
    std::fill(f.down.begin(), f.down.end(), uint8_t(0));
    const uint32_t last_row = 63 * 2 * 34;
    f.down[last_row] = 0xff; f.down[last_row + 1] = 0x7b;
    f.down[last_row + 2] = 127;
    CHECK(f.run(64, 33) == NUMERIC_OVERFLOW); CHECK(f.output == untouched);
    f.fill(64, 33); f.check(64, 33);
}

// Host-only entry point for sanitizers: allocate exact logical extents instead
// of the maximum AXI depths required by the RTL co-simulation wrapper.
static void bounds() {
    for (uint32_t hidden : {2U, 33U, 65U, MAX_HIDDEN}) {
        Fixture f;
        const uint32_t dim = hidden == MAX_HIDDEN ? MAX_DIM : 64;
        const uint32_t padded = (hidden + 31) / 32 * 32;
        f.fill(dim, hidden);
        f.gate.resize(hidden * (dim / 32) * 34); f.gate.shrink_to_fit();
        f.up.resize(f.gate.size()); f.up.shrink_to_fit();
        f.down.resize(dim * (padded / 32) * 34); f.down.shrink_to_fit();
        f.input.resize(dim); f.input.shrink_to_fit();
        f.output.resize(dim); f.output.shrink_to_fit();
        f.check(dim, hidden);
    }
    CHECK(ntrpu_ffn_q8_0(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 2) == INVALID_SHAPE);
}

static uint32_t read_u32(std::istream& file) {
    unsigned char b[4]; CHECK(bool(file.read(reinterpret_cast<char*>(b), 4)));
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

static std::vector<float> read_floats(std::istream& file, size_t count) {
    std::vector<float> values(count);
    for (auto& value : values) {
        const uint32_t bits = read_u32(file); std::memcpy(&value, &bits, sizeof(bits));
        CHECK(std::isfinite(value));
    }
    return values;
}

// Reuses NTRFFN01 generated by tools/run_ntrpu_brandon.py, including both oracles.
static void brandon(const char* path) {
    std::ifstream file(path, std::ios::binary);
    char magic[8]; CHECK(bool(file.read(magic, 8))); CHECK(std::memcmp(magic, "NTRFFN01", 8) == 0);
    const uint32_t dim = read_u32(file), hidden = read_u32(file), padded = read_u32(file), samples = read_u32(file);
    CHECK(dim != 0 && dim <= MAX_DIM && dim % 32 == 0 && hidden >= 2 && hidden <= MAX_HIDDEN);
    CHECK(padded == (hidden + 31) / 32 * 32 && samples > 0 && samples <= 256);
    Fixture f;
    CHECK(bool(file.read(reinterpret_cast<char*>(f.gate.data()), hidden * (dim / 32) * 34)));
    CHECK(bool(file.read(reinterpret_cast<char*>(f.up.data()), hidden * (dim / 32) * 34)));
    CHECK(bool(file.read(reinterpret_cast<char*>(f.down.data()), dim * (padded / 32) * 34)));
    const auto inputs = read_floats(file, samples * dim);
    const auto fp32 = read_floats(file, samples * dim);
    const auto q8 = read_floats(file, samples * dim);
    CHECK(file.peek() == std::char_traits<char>::eof());
    double squared = 0, reference_squared = 0, max_error = 0, q8_error = 0;
    for (uint32_t s = 0; s < samples; ++s) {
        std::copy_n(inputs.begin() + s * dim, dim, f.input.begin());
        CHECK(f.run(dim, hidden) == OK);
        for (uint32_t i = 0; i < dim; ++i) {
            const double value = f.output[i], ref = fp32[s * dim + i], q = q8[s * dim + i];
            CHECK(std::isfinite(value));
            const double error = value - ref, delta = std::abs(value - q);
            CHECK(delta <= 1e-6 + 1e-5 * std::abs(q));
            squared += error * error; reference_squared += ref * ref;
            max_error = std::max(max_error, std::abs(error)); q8_error = std::max(q8_error, delta);
        }
    }
    CHECK(reference_squared != 0 || squared == 0);
    const double relative = reference_squared == 0 ? 0 : std::sqrt(squared / reference_squared);
    CHECK(relative <= 0.05);
    std::cout << std::setprecision(12) << "{\"samples\":" << samples << ",\"relative_l2\":" << relative
        << ",\"max_abs_error\":" << max_error << ",\"q8_reference_max_abs_error\":" << q8_error << "}\n";
}

int main(int argc, char** argv) {
    try {
        const std::string which = argc > 1 ? argv[1] : "all";
        if (which == "--fixture") { CHECK(argc == 3); brandon(argv[2]); return 0; }
        if (argc == 2 && which != "all" && which != "numeric" && which != "invalid" &&
            which != "bounds" && which != "smoke" && which != "--all-fixture") {
            brandon(argv[1]); return 0;
        }
        if (which == "all" || which == "numeric" || which == "--all-fixture") numeric();
        // Cost-bounded RTL bank: all small numeric cases and every error case.
        // Maximum shape and real weights remain in the full bank, never implied here.
        if (which == "smoke") numeric(false);
        if (which == "all" || which == "invalid" || which == "smoke" || which == "--all-fixture") invalid();
        if (which == "bounds") bounds();
        if (which == "--all-fixture") { CHECK(argc == 3); brandon(argv[2]); }
        CHECK(which == "all" || which == "numeric" || which == "invalid" || which == "bounds" || which == "smoke" || which == "--all-fixture");
        std::cout << "PASS HLS FFN host " << which << '\n'; return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
