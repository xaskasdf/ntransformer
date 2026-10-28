#pragma once

#include <cstdint>

namespace nt { namespace ntrpu { namespace ffn_hls {
constexpr uint32_t MAX_DIM = 256;
constexpr uint32_t MAX_HIDDEN = 720;
constexpr uint32_t MAX_PADDED = 736;
constexpr uint32_t MAX_GATE_BYTES = MAX_HIDDEN * (MAX_DIM / 32) * 34;
constexpr uint32_t MAX_DOWN_BYTES = MAX_DIM * (MAX_PADDED / 32) * 34;
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;
constexpr uint32_t NONFINITE_INPUT = 2;
constexpr uint32_t INVALID_WEIGHT_SCALE = 3;
constexpr uint32_t NUMERIC_OVERFLOW = 4;
}}}

// Local FFN baseline for Brandon: dim=32..256, multiple of 32; hidden=2..720.
// Q8_0 row-major gate/up: hidden x dim; down: dim x round_up(hidden,32).
// Buffers must be valid and non-overlapping, with those exact required extents.
// Shape errors access no memory. All errors preserve the entire output buffer.
// Successful output contains dim FP32 values. Two activation quantizations per
// successful command, with reuse across gate/up tiles and across down tiles.
// This top does not implement PCIe descriptors, model epochs, DMA or DDR control.
extern "C" uint32_t ntrpu_ffn_q8_0(const uint8_t* gate, const uint8_t* up,
    const uint8_t* down, const float* input, float* output, uint32_t dim, uint32_t hidden);
