#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nt::ntrpu {

// INT8 includes -128: every partial sum fits INT32 for K <= floor(INT32_MAX/16384).
constexpr size_t GEMV_MAX_COLUMNS = 131071;
constexpr uint64_t GEMV_MAX_WEIGHT_BYTES = 1024 * 1024; // Per simulated command.

// Row-major signed INT8 W and x; returns W*x in INT32 without scaling or bias.
// Invalid shapes throw std::invalid_argument. No saturation or wraparound.
std::vector<int32_t> gemv_int8(std::span<const int8_t> weights,
                              std::span<const int8_t> input, size_t rows);

// Simulator wire representation: signed INT32, little endian.
std::vector<std::byte> pack_int32_le(std::span<const int32_t> values);
std::vector<int32_t> unpack_int32_le(std::span<const std::byte> bytes);

} // namespace nt::ntrpu
