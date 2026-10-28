#include "gemv.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace nt::ntrpu {

std::vector<int32_t> gemv_int8(std::span<const int8_t> weights,
                              std::span<const int8_t> input, size_t rows) {
    const size_t columns = input.size();
    if (rows == 0 || columns == 0 || columns > GEMV_MAX_COLUMNS ||
        rows > weights.size() / columns || rows * columns != weights.size()) {
        throw std::invalid_argument("invalid INT8 GEMV shape");
    }
    std::vector<int32_t> output(rows, 0);
    // Functional model of 32 lanes with a tail. This is not synthesized HLS.
    for (size_t row = 0; row < rows; ++row) {
        int32_t sum = 0;
        for (size_t base = 0; base < columns; base += 32) {
            int32_t partial = 0;
            const size_t end = std::min(base + 32, columns);
            for (size_t column = base; column < end; ++column) {
                partial += static_cast<int32_t>(weights[row * columns + column]) *
                           static_cast<int32_t>(input[column]);
            }
            sum += partial;
        }
        output[row] = sum;
    }
    return output;
}

std::vector<std::byte> pack_int32_le(std::span<const int32_t> values) {
    if (values.size() > std::vector<std::byte>{}.max_size() / 4) {
        throw std::length_error("INT32 byte buffer too large");
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

std::vector<int32_t> unpack_int32_le(std::span<const std::byte> bytes) {
    if (bytes.size() % 4 != 0) throw std::invalid_argument("incomplete INT32 value");
    std::vector<int32_t> output(bytes.size() / 4);
    for (size_t i = 0; i < output.size(); ++i) {
        uint32_t bits = 0;
        for (size_t byte = 0; byte < 4; ++byte) {
            bits |= std::to_integer<uint32_t>(bytes[4 * i + byte]) << (byte * 8);
        }
        output[i] = std::bit_cast<int32_t>(bits);
    }
    return output;
}

} // namespace nt::ntrpu
