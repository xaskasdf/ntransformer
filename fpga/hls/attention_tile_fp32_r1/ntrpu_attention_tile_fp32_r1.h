#pragma once
#include <cstdint>

namespace nt { namespace ntrpu { namespace attention_tile_r1 {
constexpr uint32_t HEAD_DIM = 64, TILE_TOKENS = 64, TILE_ELEMENTS = 4096;
constexpr uint32_t STATE_WORDS = 104, MAX_CONTEXT = 131072;
constexpr uint64_t STATE_MAGIC = 0x4e54524154543031ULL;
enum Status : uint32_t {
    OK = 0, INVALID_SHAPE = 1, INVALID_IDENTITY = 2,
    INVALID_STATE = 3, NONFINITE_INPUT = 4, NUMERICAL_ERROR = 5
};
}}}

// One Q head against 1..64 consecutive KV positions. FP32 Q/K/V, FP64
// online softmax state. Tokens must be submitted in increasing order from 0
// through query_position; kv_head must equal query_head/4 (Llama 1B GQA).
// All RAMs distinct, immutable inputs, state private to this query/job.
// Output state changes only on success; output changes only at the final tile.
// No AXI, ownership, cache-validity, cancellation or reset controller here.
extern "C" uint32_t ntrpu_attention_tile_fp32_r1(
    const uint32_t query[64], const uint32_t keys[4096], const uint32_t values[4096],
    const uint64_t state_input[104], uint64_t state_output[104], uint32_t output[64],
    uint32_t tile_start, uint32_t tile_count, uint32_t query_position,
    uint32_t layer, uint32_t query_head, uint32_t kv_head,
    uint64_t sequence, uint64_t epoch);
