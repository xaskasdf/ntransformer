#include "ffn.h"
#include "numeric_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace nt::ntrpu;

// Deliberately bounded fixture reader, not a model file loader.
static uint32_t read_u32(std::istream& file) {
    unsigned char bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), 4)) throw std::runtime_error("truncated header");
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

static std::vector<std::byte> read_bytes(std::istream& file, size_t size) {
    std::vector<std::byte> bytes(size);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("truncated fixture");
    }
    return bytes;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: ntrpu_ffn_fixture fixture.bin");
        std::ifstream file(std::filesystem::path(argv[1]), std::ios::binary);
        char magic[8];
        if (!file.read(magic, 8) || std::memcmp(magic, "NTRFFN01", 8) != 0) {
            throw std::runtime_error("invalid fixture magic");
        }
        const size_t dim = read_u32(file), hidden = read_u32(file);
        const size_t padded = read_u32(file), samples = read_u32(file);
        if (dim == 0 || dim % 32 != 0 || dim > 1024 || hidden < 2 || hidden > 1024 ||
            padded != (hidden + 31) / 32 * 32 || samples == 0 || samples > 256) {
            throw std::runtime_error("invalid fixture shape");
        }
        auto gate = read_bytes(file, hidden * (dim / 32) * 34);
        auto up = read_bytes(file, gate.size());
        auto down = read_bytes(file, dim * (padded / 32) * 34);
        const auto inputs = unpack_f32_le(read_bytes(file, samples * dim * 4));
        const auto fp32 = unpack_f32_le(read_bytes(file, samples * dim * 4));
        const auto quantized = unpack_f32_le(read_bytes(file, samples * dim * 4));
        if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing fixture bytes");
        double error_squared = 0, reference_squared = 0, max_error = 0, q8_max_error = 0;
        Counters counters{};
        uint64_t resident_bytes = 0;
        for (size_t s = 0; s < samples; ++s) {
            const auto result = q8_0_ffn_distributed(gate, up, down,
                std::span<const float>(inputs).subspan(s * dim, dim), hidden);
            for (size_t i = 0; i < dim; ++i) {
                const double value = result.output[i], ref = fp32[s * dim + i], q = quantized[s * dim + i];
                if (!std::isfinite(value) || !std::isfinite(ref) || !std::isfinite(q)) {
                    throw std::runtime_error("non-finite fixture/result");
                }
                const double error = value - ref, q_error = std::abs(value - q);
                error_squared += error * error;
                reference_squared += ref * ref;
                max_error = std::max(max_error, std::abs(error));
                q8_max_error = std::max(q8_max_error, q_error);
                if (q_error > 1e-6 + 1e-5 * std::abs(q)) {
                    throw std::runtime_error("distributed result differs from independent Q8 reference");
                }
            }
            counters = result.counters;
            resident_bytes = result.resident_bytes;
        }
        if (reference_squared == 0 && error_squared != 0) throw std::runtime_error("nonzero output for zero reference");
        const double relative_l2 = reference_squared == 0 ? 0 : std::sqrt(error_squared / reference_squared);
        // A fixture regression limit, not a model-quality guarantee.
        if (relative_l2 > 0.05) throw std::runtime_error("FFN relative L2 exceeds 5% fixture limit");
        std::cout << std::setprecision(12)
                  << "{\"samples\":" << samples << ",\"relative_l2\":" << relative_l2
                  << ",\"max_abs_error\":" << max_error << ",\"q8_reference_max_abs_error\":" << q8_max_error
                  << ",\"per_sample\":{\"macs\":" << counters.macs
                  << ",\"local_weight_bytes\":" << counters.local_weight_bytes
                  << ",\"peer_bytes\":" << counters.peer_bytes
                  << ",\"dma_bytes\":" << counters.bytes_copied
                  << ",\"gemv_commands\":" << counters.q8_0_commands
                  << ",\"swiglu_commands\":" << counters.swiglu_commands
                  << ",\"resident_bytes\":" << resident_bytes << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
