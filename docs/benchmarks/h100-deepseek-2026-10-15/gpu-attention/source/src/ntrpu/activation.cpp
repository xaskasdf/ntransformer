#include "activation.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nt::ntrpu {

std::vector<float> swiglu(std::span<const float> gate, std::span<const float> up) {
    if (gate.empty() || gate.size() != up.size()) throw std::invalid_argument("SwiGLU shape");
    std::vector<float> result(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i], u = up[i];
        if (!std::isfinite(g) || !std::isfinite(u)) throw std::invalid_argument("SwiGLU operand");
        const double e = std::exp(-std::abs(g));
        const double sigmoid = g >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
        const double value = (g * sigmoid) * u;
        if (!std::isfinite(value) || std::abs(value) > double(std::numeric_limits<float>::max())) {
            throw std::invalid_argument("SwiGLU output");
        }
        result[i] = static_cast<float>(value);
    }
    return result;
}

} // namespace nt::ntrpu
