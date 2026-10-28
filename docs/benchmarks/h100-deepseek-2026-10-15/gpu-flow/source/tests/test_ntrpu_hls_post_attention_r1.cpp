#include "ntrpu_post_attention_r1.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nt::ntrpu::post_attention_r1;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #x); } while (false)
static uint32_t read32(std::istream& f) {
    unsigned char b[4]; f.read(reinterpret_cast<char*>(b), 4); CHECK(bool(f));
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}
static uint64_t read64(std::istream& f) {
    const uint32_t low = read32(f), high = read32(f);
    return low | (uint64_t(high) << 32);
}
int main(int argc, char** argv) {
    try {
        CHECK(argc == 2); std::ifstream file(argv[1], std::ios::binary); CHECK(bool(file));
        char magic[8]; file.read(magic, 8); CHECK(std::memcmp(magic, "NTRPOST1", 8) == 0);
        const uint32_t count = read32(file); CHECK(count > 0 && count < 256);
        uint32_t successful = 0, residual_words = 0, abi_words = 0;
        for (uint32_t id = 1; id <= count; ++id) {
            CHECK(read32(file) == id);
            const uint32_t n = read32(file), eps = read32(file), want = read32(file);
            CHECK(want <= QUANTIZE_ERROR);
            CHECK(want != OK || (n > 0 && n <= MAX_ELEMENTS && n % 32 == 0));
            std::array<uint32_t, MAX_ELEMENTS> x, branch, weight, expected_r;
            std::array<uint64_t, MAX_WORDS> expected_q;
            for (auto& v : x) v = read32(file);
            for (auto& v : branch) v = read32(file);
            for (auto& v : weight) v = read32(file);
            if (want == OK) {
                for (uint32_t i = 0; i < n; ++i) expected_r[i] = read32(file);
                for (uint32_t i = 0; i < n / 4; ++i) expected_q[i] = read64(file);
            }
            const auto old_x = x, old_branch = branch, old_weight = weight;
            std::array<uint32_t, MAX_ELEMENTS + 2> r;
            std::array<uint64_t, MAX_WORDS + 2> q;
            r.fill(0xa55a3cc3U); q.fill(0xa55a5aa53cc3c33cULL);
            const auto old_r = r; const auto old_q = q;
            const uint32_t status = ntrpu_post_attention_r1(x.data(), branch.data(), weight.data(),
                r.data() + 1, q.data() + 1, n, eps);
            CHECK(status == want && x == old_x && branch == old_branch && weight == old_weight);
            CHECK(r.front() == old_r.front() && r.back() == old_r.back());
            CHECK(q.front() == old_q.front() && q.back() == old_q.back());
            if (status == OK) {
                for (uint32_t i = 0; i < n; ++i) {
                    CHECK(r[i + 1] == expected_r[i]);
                    std::cout << "POST_R case=" << id << " index=" << i << " bits="
                        << std::hex << std::setw(8) << std::setfill('0') << r[i + 1] << std::dec << '\n';
                }
                for (uint32_t i = 0; i < n / 4; ++i) {
                    CHECK(q[i + 1] == expected_q[i]);
                    std::cout << "POST_Q case=" << id << " index=" << i << " bits="
                        << std::hex << std::setw(16) << std::setfill('0') << q[i + 1] << std::dec << '\n';
                }
                for (uint32_t i = n; i < MAX_ELEMENTS; ++i) CHECK(r[i + 1] == old_r[i + 1]);
                for (uint32_t i = n / 4; i < MAX_WORDS; ++i) CHECK(q[i + 1] == old_q[i + 1]);
                ++successful; residual_words += n; abi_words += n / 4;
            } else CHECK(r == old_r && q == old_q);
            std::cout << "POST_CASE id=" << id << " elements=" << n << " epsilon="
                << std::hex << std::setw(8) << std::setfill('0') << eps << std::dec
                << " status=" << status << " residual=" << (status == OK ? n : 0)
                << " abi=" << (status == OK ? n / 4 : 0) << '\n';
        }
        CHECK(file.peek() == std::char_traits<char>::eof());
        std::cout << "POST_PASS cases=" << count << " successful=" << successful
            << " residual=" << residual_words << " abi=" << abi_words << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
