#pragma once

#include <cstdint>

namespace nt {
namespace ntrpu {
namespace hls {

constexpr uint32_t MAX_ROWS = 32;
constexpr uint32_t MAX_COLUMNS = 1024;
constexpr uint32_t MAX_WEIGHTS = MAX_ROWS * MAX_COLUMNS;
constexpr uint32_t LANES = 32;
constexpr uint32_t OK = 0;
constexpr uint32_t INVALID_SHAPE = 1;

} // namespace hls
} // namespace ntrpu
} // namespace nt

// Standalone synthesis top. Row-major, tightly packed INT8 operands; INT32 output.
// Caller supplies valid, non-overlapping memory regions of the required lengths.
// Invalid dimensions return INVALID_SHAPE without accessing any memory port.
extern "C" uint32_t ntrpu_gemv_int8(const int8_t* weights, const int8_t* input,
                                    int32_t* output, uint32_t rows, uint32_t columns);
