#pragma once

#include <span>
#include <vector>

namespace nt::ntrpu {

// Host numeric model: double sigmoid/product, one final FP32 rounding.
// Throws invalid_argument for shape errors or non-finite/unrepresentable output.
std::vector<float> swiglu(std::span<const float> gate, std::span<const float> up);

} // namespace nt::ntrpu
