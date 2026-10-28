// Full-size numerical test; real Q8 weights and synthetic activations.
// This executable exercises kernel C++ source, not RTL or full inference.
#include "ntrpu_q8_block_mac.h"
#include "ntrpu_quantize_q8_0.h"
#include "ntrpu_swiglu.h"
#include "ntrpu_swiglu_q8_block.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t D = 2048, H = 8192;
using Bytes = std::vector<uint8_t>;
static_assert(std::endian::native == std::endian::little);

template<class T>
std::vector<T> read_exact(const std::filesystem::path& path, size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(T)) {
        throw std::runtime_error("File size: " + path.filename().string());
    }
    std::vector<T> result(count);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(result.data()), count * sizeof(T))) {
        throw std::runtime_error("Read: " + path.filename().string());
    }
    return result;
}

void equal_bytes(const std::string& label, const void* actual, const void* expected, size_t count) {
    const auto* a = static_cast<const uint8_t*>(actual);
    const auto* e = static_cast<const uint8_t*>(expected);
    for (size_t index = 0; index < count; ++index) {
        if (a[index] != e[index]) {
            throw std::runtime_error(label + " byte=" + std::to_string(index) +
                                     " actual=" + std::to_string(a[index]) +
                                     " expected=" + std::to_string(e[index]));
        }
    }
}

Bytes quantize(const std::vector<float>& input) {
    Bytes result(input.size() / 32 * 64, 0);
    for (size_t block = 0; block < input.size() / 32; ++block) {
        std::array<int8_t, 32> q{};
        double scale = 0;
        if (ntrpu_quantize_q8_0(input.data() + block * 32, q.data(), &scale, 32) != 0) {
            throw std::runtime_error("Quantization status");
        }
        std::memcpy(result.data() + block * 64, q.data(), 32);
        std::memcpy(result.data() + block * 64 + 32, &scale, 8);
    }
    return result;
}

std::vector<double> project(const Bytes& weights, size_t rows, const Bytes& activation) {
    const size_t blocks = activation.size() / 64;
    std::vector<double> result(rows, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            const uint8_t* weight = weights.data() + (row * blocks + block) * 34;
            const uint8_t* input = activation.data() + block * 64;
            std::array<uint64_t, 4> w{}, x{};
            uint16_t scale = 0;
            double input_scale = 0, value = 0;
            std::memcpy(&scale, weight, 2);
            std::memcpy(w.data(), weight + 2, 32);
            std::memcpy(x.data(), input, 32);
            std::memcpy(&input_scale, input + 32, 8);
            if (ntrpu_q8_block_mac(w[0], w[1], w[2], w[3], x[0], x[1], x[2], x[3],
                                  scale, input_scale, result[row], &value) != 0) {
                throw std::runtime_error("MAC status");
            }
            result[row] = value;
        }
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::runtime_error("Expected FIXTURE [--inject-bad-reference]");
        const bool mutate = argc == 3 && std::string(argv[2]) == "--inject-bad-reference";
        if (argc == 3 && !mutate) throw std::runtime_error("Unknown option");
        const std::filesystem::path folder = argv[1];
        const auto gate_weight = read_exact<uint8_t>(folder / "weight-1-dense.bin", H * D / 32 * 34);
        const auto up_weight = read_exact<uint8_t>(folder / "weight-2-dense.bin", H * D / 32 * 34);
        const auto down_weight = read_exact<uint8_t>(folder / "weight-3-dense.bin", D * H / 32 * 34);
        for (int sample = 1; sample <= 3; ++sample) {
            const auto start = std::chrono::steady_clock::now();
            const std::string prefix = "job-00" + std::to_string(sample);
            auto path = [&](const std::string& suffix) { return folder / (prefix + "-" + suffix + ".bin"); };
            const auto x = read_exact<float>(path("input-fp32"), D);
            const auto input = quantize(x);
            const auto input_ref = read_exact<uint8_t>(path("input-abi"), D / 32 * 64);
            equal_bytes("input ABI", input.data(), input_ref.data(), input.size());
            const auto gate = project(gate_weight, H, input);
            const auto up = project(up_weight, H, input);
            auto gate_ref = read_exact<double>(path("gate-fp64"), H);
            const auto up_ref = read_exact<double>(path("up-fp64"), H);
            if (mutate) reinterpret_cast<uint8_t*>(gate_ref.data())[0] ^= 1;
            equal_bytes("gate", gate.data(), gate_ref.data(), H * 8);
            equal_bytes("up", up.data(), up_ref.data(), H * 8);
            Bytes activation(H / 32 * 64, 0);
            std::vector<float> activated(H, 0);
            for (size_t block = 0; block < H / 32; ++block) {
                std::array<uint64_t, 32> gb{}, ub{};
                std::array<uint64_t, 8> record{};
                std::array<float, 32> gf{}, uf{};
                for (size_t lane = 0; lane < 32; ++lane) {
                    const size_t index = block * 32 + lane;
                    std::memcpy(&gb[lane], &gate[index], 8);
                    std::memcpy(&ub[lane], &up[index], 8);
                    gf[lane] = static_cast<float>(gate[index]);
                    uf[lane] = static_cast<float>(up[index]);
                }
                if (ntrpu_swiglu(gf.data(), uf.data(), activated.data() + block * 32, 32) != 0 ||
                    ntrpu_swiglu_q8_block(gb.data(), ub.data(), record.data(), 32) != 0) {
                    throw std::runtime_error("SwiGLU status");
                }
                std::memcpy(activation.data() + block * 64, record.data(), 64);
            }
            const auto activated_ref = read_exact<float>(path("activation-fp32"), H);
            const auto activation_ref = read_exact<uint8_t>(path("activation-abi"), H / 32 * 64);
            equal_bytes("SwiGLU FP32", activated.data(), activated_ref.data(), H * 4);
            equal_bytes("SwiGLU ABI", activation.data(), activation_ref.data(), activation.size());
            const auto separately_quantized = quantize(activated);
            equal_bytes("composed SwiGLU/quantize ABI", activation.data(), separately_quantized.data(), activation.size());
            const auto down = project(down_weight, D, activation);
            const auto down_ref = read_exact<double>(path("down-fp64"), D);
            equal_bytes("down", down.data(), down_ref.data(), D * 8);
            std::vector<float> final(D, 0);
            for (size_t index = 0; index < D; ++index) final[index] = static_cast<float>(down[index]);
            const auto final_ref = read_exact<float>(path("final-fp32"), D);
            equal_bytes("final", final.data(), final_ref.data(), D * 4);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "NTRPU_LLAMA32_FFN_NATIVE_CASE sample=" << sample
                      << " dim=2048 hidden=8192 mac_blocks=1572864 reference_values=28672"
                      << " activation_records=256 bit_exact=1 cpu_seconds=" << seconds << '\n';
        }
#ifdef NTRPU_USE_HLS_MATH
        constexpr const char* math = "hls";
#else
        constexpr const char* math = "stdlib";
#endif
        std::cout << "NTRPU_LLAMA32_FFN_NATIVE_PASS samples=3 reference_values=86016 mac_blocks=4718592 math=" << math << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NTRPU_LLAMA32_FFN_NATIVE_FAIL " << error.what() << '\n';
        return 1;
    }
}
