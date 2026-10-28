#include "ntrpu_q8_block_mac.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::block_mac;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + \
    std::to_string(__LINE__) + ": " #c); } while (false)

static unsigned transactions = 0;
struct Block {
    std::array<int, 32> w{}, x{};
    uint16_t bits = 0x3c00;
    double scale = 1;
    static std::array<uint64_t, 4> pack(const std::array<int, 32>& v) {
        std::array<uint64_t, 4> p{};
        for (unsigned i = 0; i < 32; ++i) p[i / 8] |= uint64_t(uint8_t(v[i])) << (8 * (i % 8));
        return p;
    }
    uint32_t run(double prior, double& out) const {
        const auto a = pack(w), b = pack(x);
        ++transactions;
        return ntrpu_q8_block_mac(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], bits, scale, prior, &out);
    }
    double contribution() const {
        int64_t dot = 0;
        for (unsigned i = 0; i < 32; ++i) dot += int64_t(w[i]) * x[i];
        const unsigned e = (bits >> 10) & 31, f = bits & 1023;
        double s = e ? std::ldexp(double(1024 + f), int(e) - 25) : std::ldexp(double(f), -24);
        if (bits & 0x8000) s = -s;
        return (s * scale) * double(dot);
    }
    void check(double prior) const {
        double output = 1234567.25;
        CHECK(run(prior, output) == OK);
        const double expected = prior + contribution();
        CHECK(output == expected);
    }
};

static void numeric() {
    Block b;
    // Each byte position has its own signed test, including the word boundaries.
    for (unsigned lane = 0; lane < 32; ++lane) {
        b.w.fill(0); b.x.fill(-128); b.w[lane] = lane & 1 ? 127 : -128;
        b.check(0);
    }
    b.w.fill(-128); b.x.fill(-128); b.check(0);
    b.w.fill(127); b.x.fill(-128); b.check(-65536);
    for (uint16_t bits : {0U, 0x8000U, 1U, 0x8001U, 0x3ffU, 0x400U, 0x3c00U, 0xbc00U, 0x7bffU, 0xfbffU}) {
        b.bits = bits; b.scale = 1.0 / 127; b.check(8.5);
    }
    b.bits = 0x7bff; b.scale = double(std::numeric_limits<float>::max()) / 127.0; b.check(0);
    b.bits = 1; b.scale = double(std::numeric_limits<float>::denorm_min()) / 127.0; b.check(0);
    b.scale = 0; b.check(-17);

    // One row with Llama K=8192, eight K tiles of 1024. Every block feeds the
    // previous accumulator; exact scalar reference detects reset/reassociation.
    uint32_t seed = 1397;
    auto next = [&seed]() { seed = seed * 1664525U + 1013904223U; return seed >> 16; };
    double sum = 0, reference = 0;
    for (unsigned tile = 0; tile < 8; ++tile) {
        for (unsigned block = 0; block < 32; ++block) {
            for (unsigned lane = 0; lane < 32; ++lane) {
                b.w[lane] = int(next() % 256) - 128; b.x[lane] = int(next() % 256) - 128;
            }
            b.bits = uint16_t((next() % 0x7c00) | ((block & 1) << 15));
            b.scale = (next() % 1000) / 127.0;
            reference += b.contribution();
            double output = 0; CHECK(b.run(sum, output) == OK); sum = output;
            CHECK(sum == reference);
        }
    }
    // A known non-associative sequence across a tile boundary: a subtotal
    // implementation produces 1, whereas the required sequential sum is 0.
    b.w.fill(0); b.x.fill(0); b.w[0] = 1; b.x[0] = 1;
    b.bits = 0x3c00; b.scale = 1;
    double out = 0; CHECK(b.run(9007199254740992.0, out) == OK);
    CHECK(out == 9007199254740992.0);
    b.bits = 0xbc00; b.scale = 9007199254740992.0;
    double last = 1; CHECK(b.run(out, last) == OK); CHECK(last == 0);
}

static void invalid() {
    Block b; b.w.fill(127); b.x.fill(127);
    for (uint16_t bits : {0x7c00U, 0xfc00U, 0x7e00U, 0xffffU}) {
        b.bits = bits; double out = 123;
        CHECK(b.run(0, out) == INVALID_SCALE); CHECK(out == 123);
    }
    b.bits = 0x3c00;
    for (double scale : {-1.0, std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(), double(std::numeric_limits<float>::max())}) {
        b.scale = scale; double out = -345;
        CHECK(b.run(0, out) == INVALID_SCALE); CHECK(out == -345);
    }
    b.scale = 1;
    for (double prior : {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
        double out = 456;
        CHECK(b.run(prior, out) == INVALID_ACCUMULATOR); CHECK(out == 456);
    }
    b.check(0); // recovery has no stale internal accumulator
}

static void exhaustive() {
    Block b; b.w[0] = 1; b.x[0] = 1;
    for (unsigned bits = 0; bits <= 65535; ++bits) {
        b.bits = uint16_t(bits);
        if ((bits & 0x7c00) == 0x7c00) {
            double out = 123; CHECK(b.run(0, out) == INVALID_SCALE); CHECK(out == 123);
        } else b.check(0);
    }
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 2 ? argv[1] : "all";
        CHECK(argc <= 2);
        if (mode == "all") { numeric(); invalid(); }
        else if (mode == "exhaustive") exhaustive();
        else throw std::runtime_error("unknown mode");
        std::cout << "PASS Q8_BLOCK_MAC mode=" << mode << " transactions=" << transactions << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL Q8_BLOCK_MAC " << e.what() << '\n'; return 1;
    }
}
