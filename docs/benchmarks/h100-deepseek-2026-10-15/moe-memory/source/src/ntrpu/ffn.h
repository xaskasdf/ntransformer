#pragma once

#include "simulator.h"

namespace nt::ntrpu {

struct FfnResult {
    std::vector<float> output;
    Counters counters;
    uint64_t resident_bytes;
};

// One token, no norm/residual/dropout. Gate/up rows split over two simulated units;
// down runs on unit 1. Internal values move only through local compute and DMA.
// D=input.size() is a multiple of 32; gate/up are H x D; down is D x ceil32(H).
// The caller zero-pads down's last columns. H >= 2; each GEMV fits its 1 MiB budget.
FfnResult q8_0_ffn_distributed(std::span<const std::byte> gate,
                             std::span<const std::byte> up,
                             std::span<const std::byte> down,
                             std::span<const float> input, size_t hidden);

} // namespace nt::ntrpu
