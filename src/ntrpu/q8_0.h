#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nt::ntrpu {

constexpr size_t Q8_0_BLOCK_ELEMENTS = 32;
constexpr size_t Q8_0_BLOCK_BYTES = 34;

// Experimental host validation, not a GGUF file loader or hardware runtime.
// Weights: tightly packed little-endian Q8_0 blocks, row-major, no padding.
// Input length must be a positive multiple of 32, <= GEMV_MAX_COLUMNS.
// Malformed shapes and non-finite scales/activations throw invalid_argument.
std::vector<double> q8_0_gemv_reference(std::span<const std::byte> weights,
                                      std::span<const float> input, size_t rows);

// Quantize activations per block to [-127,127], ties away from zero; scale=maxabs/127.
// Up to 1024 columns: calls the activation quantizer and scaled Q8_0 HLS sources as C++.
// Larger inputs: INT8 top on 32-column tiles, scaling/accumulation in the host adapter.
// Both use double to isolate quantization error from rounding. Neither is RTL execution.
std::vector<double> q8_0_gemv_hls_model(std::span<const std::byte> weights,
                                      std::span<const float> input, size_t rows);

} // namespace nt::ntrpu
