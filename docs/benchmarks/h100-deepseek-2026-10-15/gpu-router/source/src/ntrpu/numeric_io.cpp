#include "numeric_io.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace nt::ntrpu {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "NTRPU numeric transport requires IEEE-754 binary32");

std::vector<std::byte> pack_f32_le(std::span<const float> values) {
    if (values.size() > std::vector<std::byte>{}.max_size() / 4) {
        throw std::length_error("FP32 byte buffer too large");
    }
    std::vector<std::byte> output(values.size() * 4);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint32_t bits = std::bit_cast<uint32_t>(values[i]);
        for (size_t byte = 0; byte < 4; ++byte) {
            output[4 * i + byte] = static_cast<std::byte>((bits >> (byte * 8)) & 0xffU);
        }
    }
    return output;
}

std::vector<float> unpack_f32_le(std::span<const std::byte> bytes) {
    if (bytes.size() % 4 != 0) throw std::invalid_argument("incomplete FP32 value");
    std::vector<float> output(bytes.size() / 4);
    for (size_t i = 0; i < output.size(); ++i) {
        uint32_t bits = 0;
        for (size_t byte = 0; byte < 4; ++byte) {
            bits |= std::to_integer<uint32_t>(bytes[4 * i + byte]) << (byte * 8);
        }
        output[i] = std::bit_cast<float>(bits);
    }
    return output;
}

} // namespace nt::ntrpu
