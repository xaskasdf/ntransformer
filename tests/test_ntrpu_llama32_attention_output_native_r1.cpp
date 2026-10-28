// Actual kernel C++ composition, using four real layer-0 attention outputs.
// Native arithmetic evidence only; this is not an RTL/AXI/inference benchmark.
#include "ntrpu_q8_block_mac.h"
#include "ntrpu_quantize_q8_0.h"
#include "ntrpu_swiglu.h"
#include "ntrpu_residual_fp32_r1.h"
#include "ntrpu_norm_q8_abi1_r1.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t D = 2048;
using Bytes = std::vector<uint8_t>;

template<class T>
std::vector<T> read_exact(const std::string& path, size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streampos(count * sizeof(T)))
        throw std::runtime_error("File size: " + path);
    std::vector<T> result(count);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(result.data()), count * sizeof(T)))
        throw std::runtime_error("Read: " + path);
    return result;
}

void equal_bytes(const std::string& label, const void* actual, const void* expected, size_t count) {
    const auto* a = static_cast<const uint8_t*>(actual);
    const auto* e = static_cast<const uint8_t*>(expected);
    for (size_t i = 0; i < count; ++i)
        if (a[i] != e[i]) throw std::runtime_error(label + " byte=" + std::to_string(i));
}

void emit(const std::string& path, const void* data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.write(static_cast<const char*>(data), bytes)) throw std::runtime_error("Write: " + path);
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("Expected FIXTURE OUTPUT [--inject-bad-reference]");
        const bool mutate = argc == 4 && std::string(argv[3]) == "--inject-bad-reference";
        if (argc == 4 && !mutate) throw std::runtime_error("Unknown option");
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("Little endian required");
        const std::string folder = argv[1], output = argv[2];
        const auto weights = read_exact<uint8_t>(folder + "/output-weight-image.bin", 4032 + D * D / 32 * 34);
        const auto norm_weights = read_exact<uint32_t>(folder + "/ffn-norm-weight.bin", D);
        for (size_t sample = 1; sample <= 4; ++sample) {
            const std::string prefix = "sample-00" + std::to_string(sample);
            auto path = [&](const std::string& suffix) { return folder + "/" + prefix + "-" + suffix + ".bin"; };
            const auto input = read_exact<float>(path("attention-fp32"), D);
            Bytes activation(D / 32 * 64, 0);
            for (size_t block = 0; block < D / 32; ++block) {
                std::array<int8_t, 32> q{}; double scale = 0;
                if (ntrpu_quantize_q8_0(input.data() + block * 32, q.data(), &scale, 32) != 0)
                    throw std::runtime_error("Quantization status");
                std::memcpy(activation.data() + block * 64, q.data(), 32);
                std::memcpy(activation.data() + block * 64 + 32, &scale, 8);
            }
            const auto abi_ref = read_exact<uint8_t>(path("attention-abi1"), activation.size());
            equal_bytes("Attention ABI", activation.data(), abi_ref.data(), activation.size());
            std::vector<double> projection(D, 0.0);
            // The memory-image layout is group32 / Ktile32 / Kblock / row / Q8_0.
            for (size_t group = 0; group < D / 32; ++group) {
                for (size_t tile = 0; tile < 2; ++tile) {
                    for (size_t block = 0; block < 32; ++block) {
                        const auto* x = activation.data() + (tile * 32 + block) * 64;
                        std::array<uint64_t, 4> xb{}; double xs = 0;
                        std::memcpy(xb.data(), x, 32); std::memcpy(&xs, x + 32, 8);
                        for (size_t lane = 0; lane < 32; ++lane) {
                            const size_t row = group * 32 + lane;
                            const auto* w = weights.data() + 4032 + (group * 2 + tile) * 34816 + (block * 32 + lane) * 34;
                            uint16_t ws = 0; std::array<uint64_t, 4> wb{}; double value = 0;
                            std::memcpy(&ws, w, 2); std::memcpy(wb.data(), w + 2, 32);
                            if (ntrpu_q8_block_mac(wb[0], wb[1], wb[2], wb[3], xb[0], xb[1], xb[2], xb[3],
                                                  ws, xs, projection[row], &value) != 0)
                                throw std::runtime_error("MAC status");
                            projection[row] = value;
                        }
                    }
                }
            }
            auto projection_ref = read_exact<double>(path("output-fp64"), D);
            if (mutate) reinterpret_cast<uint8_t*>(projection_ref.data())[0] ^= 1;
            equal_bytes("O FP64", projection.data(), projection_ref.data(), D * 8);
            std::vector<uint32_t> branch(D), residual(D);
            for (size_t i = 0; i < D; ++i) {
                float value;
                if (!nt::ntrpu::swiglu_hls::detail::narrow(projection[i], value)) throw std::runtime_error("Narrow status");
                std::memcpy(&branch[i], &value, 4);
            }
            const auto branch_ref = read_exact<uint32_t>(path("output-fp32"), D);
            equal_bytes("O FP32", branch.data(), branch_ref.data(), D * 4);
            const auto embedding = read_exact<uint32_t>(path("embedding-fp32"), D);
            if (ntrpu_residual_fp32_r1(embedding.data(), branch.data(), residual.data(), D) != 0)
                throw std::runtime_error("Residual status");
            const auto residual_ref = read_exact<uint32_t>(path("residual-fp32"), D);
            equal_bytes("Residual", residual.data(), residual_ref.data(), D * 4);
            std::vector<uint64_t> ffn(D / 4, 0);
            if (ntrpu_norm_q8_abi1_r1(residual.data(), norm_weights.data(), ffn.data(), D, 0x3727c5acU) != 0)
                throw std::runtime_error("FFN norm status");
            const auto ffn_ref = read_exact<uint64_t>(path("ffn-input-abi1"), D / 4);
            equal_bytes("FFN ABI", ffn.data(), ffn_ref.data(), D * 2);
            emit(output + "/" + prefix + "-attention-abi1.bin", activation.data(), D * 2);
            emit(output + "/" + prefix + "-output-fp64.bin", projection.data(), D * 8);
            emit(output + "/" + prefix + "-output-fp32.bin", branch.data(), D * 4);
            emit(output + "/" + prefix + "-residual-fp32.bin", residual.data(), D * 4);
            emit(output + "/" + prefix + "-ffn-input-abi1.bin", ffn.data(), D * 2);
            std::cout << "ATTENTION_OUTPUT_NATIVE_CASE sample=" << sample
                      << " mac_blocks=131072 projection=2048 residual=2048 abi_bytes=8192 bit_exact=1\n";
        }
        std::cout << "ATTENTION_OUTPUT_NATIVE_PASS samples=4 mac_blocks=524288 projection=8192 residual=8192 math=hls\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ATTENTION_OUTPUT_NATIVE_FAIL " << error.what() << '\n';
        return 1;
    }
}
