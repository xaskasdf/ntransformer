#include "q8_0.h"
#include "gemv.h"
#include "ntrpu_gemv_int8.h"
#include "ntrpu_gemv_q8_0.h"
#include "ntrpu_quantize_q8_0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nt::ntrpu {

static double block_scale(std::span<const std::byte> weights, size_t offset) {
    const uint32_t bits = std::to_integer<uint32_t>(weights[offset]) |
                         (std::to_integer<uint32_t>(weights[offset + 1]) << 8);
    const int exponent = static_cast<int>((bits >> 10) & 31U);
    const int mantissa = static_cast<int>(bits & 1023U);
    if (exponent == 31) throw std::invalid_argument("non-finite Q8_0 scale");
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -24) :
                                            std::ldexp(double(1024 + mantissa), exponent - 25);
    return (bits & 0x8000U) ? -magnitude : magnitude;
}

static int8_t signed_weight(std::byte byte) {
    const int value = std::to_integer<int>(byte);
    return static_cast<int8_t>(value < 128 ? value : value - 256);
}

static size_t validate(std::span<const std::byte> weights, std::span<const float> input, size_t rows) {
    if (rows == 0 || input.empty() || input.size() % Q8_0_BLOCK_ELEMENTS != 0 ||
        input.size() > GEMV_MAX_COLUMNS) {
        throw std::invalid_argument("invalid Q8_0 GEMV shape");
    }
    const size_t blocks = input.size() / Q8_0_BLOCK_ELEMENTS;
    const size_t row_bytes = blocks * Q8_0_BLOCK_BYTES;
    if (rows != weights.size() / row_bytes || weights.size() % row_bytes != 0) {
        throw std::invalid_argument("Q8_0 byte count does not match dimensions");
    }
    for (float value : input) {
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite activation");
    }
    for (size_t offset = 0; offset < weights.size(); offset += Q8_0_BLOCK_BYTES) {
        block_scale(weights, offset);
    }
    return blocks;
}

std::vector<double> q8_0_gemv_reference(std::span<const std::byte> weights,
                                      std::span<const float> input, size_t rows) {
    const size_t blocks = validate(weights, input, rows);
    std::vector<double> output(rows, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks; ++block) {
            const size_t offset = (row * blocks + block) * Q8_0_BLOCK_BYTES;
            const double scale = block_scale(weights, offset);
            // Dequantize each weight independently; do not reuse the integer MAC path.
            for (size_t j = 0; j < Q8_0_BLOCK_ELEMENTS; ++j) {
                const double weight = scale * signed_weight(weights[offset + 2 + j]);
                output[row] += weight * double(input[block * Q8_0_BLOCK_ELEMENTS + j]);
            }
        }
    }
    return output;
}

std::vector<double> q8_0_gemv_hls_model(std::span<const std::byte> weights,
                                      std::span<const float> input, size_t rows) {
    const size_t blocks = validate(weights, input, rows);
    std::vector<double> output(rows, 0);
    if (input.size() <= q8_hls::MAX_COLUMNS) {
        // Feed complete rows to the scaled HLS top, including every Q8_0 block.
        // Both candidate HLS tops execute here as ordinary C++, not RTL.
        std::array<int8_t, q8_hls::MAX_COLUMNS> quantized{};
        std::array<double, q8_hls::MAX_BLOCKS> scales{};
        std::array<uint8_t, q8_hls::MAX_WEIGHT_BYTES> packed{};
        std::array<double, q8_hls::MAX_ROWS> tile_output{};
        static_assert(q8_hls::MAX_COLUMNS == quantize_hls::MAX_COLUMNS &&
                      q8_hls::BLOCK_ELEMENTS == quantize_hls::BLOCK_ELEMENTS,
                      "Quantizer and GEMV must share activation layout");
        const uint32_t quantize_status = ntrpu_quantize_q8_0(input.data(), quantized.data(), scales.data(),
            static_cast<uint32_t>(input.size()));
        if (quantize_status != quantize_hls::OK) throw std::logic_error("validated Q8_0 activation rejected");
        const size_t row_bytes = blocks * Q8_0_BLOCK_BYTES;
        for (size_t first = 0; first < rows; first += q8_hls::MAX_ROWS) {
            const size_t count = std::min(size_t{q8_hls::MAX_ROWS}, rows - first);
            for (size_t byte = 0; byte < count * row_bytes; ++byte) {
                packed[byte] = std::to_integer<uint8_t>(weights[first * row_bytes + byte]);
            }
            const uint32_t status = ntrpu_gemv_q8_0(packed.data(), quantized.data(), scales.data(),
                tile_output.data(), static_cast<uint32_t>(count), static_cast<uint32_t>(input.size()));
            if (status != q8_hls::OK) throw std::logic_error("validated scaled Q8_0 tile rejected");
            std::copy_n(tile_output.begin(), count, output.begin() + first);
        }
        return output;
    }
    // Full declared interface depths also keep this host adapter easy to inspect.
    std::array<int8_t, hls::MAX_WEIGHTS> tile{};
    std::array<int8_t, hls::MAX_COLUMNS> activation{};
    std::array<int32_t, hls::MAX_ROWS> partial{};
    for (size_t block = 0; block < blocks; ++block) {
        double maximum = 0;
        for (size_t j = 0; j < Q8_0_BLOCK_ELEMENTS; ++j) {
            maximum = std::max(maximum, std::abs(double(input[block * Q8_0_BLOCK_ELEMENTS + j])));
        }
        const double scale_x = maximum / 127.0;
        for (size_t j = 0; j < Q8_0_BLOCK_ELEMENTS; ++j) {
            const double value = scale_x == 0 ? 0 : double(input[block * Q8_0_BLOCK_ELEMENTS + j]) / scale_x;
            activation[j] = static_cast<int8_t>(std::clamp(std::round(value), -127.0, 127.0));
        }
        for (size_t first_row = 0; first_row < rows; first_row += hls::MAX_ROWS) {
            const size_t tile_rows = std::min(size_t{hls::MAX_ROWS}, rows - first_row);
            for (size_t row = 0; row < tile_rows; ++row) {
                const size_t offset = ((first_row + row) * blocks + block) * Q8_0_BLOCK_BYTES;
                for (size_t j = 0; j < Q8_0_BLOCK_ELEMENTS; ++j) {
                    tile[row * Q8_0_BLOCK_ELEMENTS + j] = signed_weight(weights[offset + 2 + j]);
                }
            }
            const auto status = ntrpu_gemv_int8(tile.data(), activation.data(), partial.data(),
                static_cast<uint32_t>(tile_rows), static_cast<uint32_t>(Q8_0_BLOCK_ELEMENTS));
            if (status != hls::OK) throw std::logic_error("validated Q8_0 tile rejected");
            for (size_t row = 0; row < tile_rows; ++row) {
                const size_t offset = ((first_row + row) * blocks + block) * Q8_0_BLOCK_BYTES;
                output[first_row + row] += block_scale(weights, offset) * scale_x * partial[row];
            }
        }
    }
    return output;
}

} // namespace nt::ntrpu
