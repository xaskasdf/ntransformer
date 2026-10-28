#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nt::ntrpu {

// IEEE-754 binary32, little endian. Serialization preserves all bit patterns;
// numerical validity is checked by the operation consuming the values.
std::vector<std::byte> pack_f32_le(std::span<const float> values);
std::vector<float> unpack_f32_le(std::span<const std::byte> bytes);

} // namespace nt::ntrpu
