#pragma once

// ============================================================
// GLM-5.3-Flash GPU offload (P1a): resident backbone projections
// + LRU-cached routed experts in VRAM. Host-driven, ds41-style.
// ============================================================

#include <cstdint>
#include <functional>

namespace nt {
namespace glm5 {

bool glm5_gpu_enabled();

// One-time device upload keyed by host pointer; returns device pointer.
const void* glm5_resident(const void* host, size_t bytes);

// y[rows] = W @ x, host in/out. W must be Q8_0 [rows, cols] or F32.
bool glm5_q8_proj(const void* w, const float* x, float* y, int64_t rows, int64_t cols);
bool glm5_f32_proj(const void* w, const float* x, float* y, int64_t rows, int64_t cols);

// Many same-input GEMVs with one upload + one sync: mats[i] is Q8_0 [rows[i], cols],
// outputs concatenated into y (offsets precomputed by caller or sequential).
bool glm5_q8_proj_many(const void* const* mats, const int64_t* rows, int n_mats,
                       int64_t cols, const float* x, float* y);

// Batched over T tokens: y[t*rows + r] = W @ x[t*cols ..], host in/out.
bool glm5_q8_proj_tokens(const void* w, const float* x, float* y,
                         int64_t rows, int64_t cols, int T);
bool glm5_f32_proj_tokens(const void* w, const float* x, float* y,
                          int64_t rows, int64_t cols, int T);

// Batched per-head over T tokens: x is [T, groups, cols], y is [T, groups, rows].
bool glm5_q8_proj_grouped_t(const void* w, const float* x, float* y,
                            int64_t rows, int64_t cols, int groups, int T);

// Batched routed experts with a VRAM LRU cache.
// gate/up: [n_expert, ff, hidden] Q8_0 ; down: [n_expert, hidden, ff] Q8_0.
// out[hidden] = sum_k weights[k] * down_k(silu_clamp(gate_k x, up_k x)).
// host_work runs on the caller thread while the GPU stream is in flight.
bool glm5_gpu_experts(const float* x, int64_t hidden,
                      const void* gate_base, const void* up_base, const void* down_base,
                      int64_t ff, const int* experts, const float* weights, int n_used,
                      float limit, float* out,
                      const std::function<void()>& host_work = std::function<void()>());

// LM head, Q8_0 [vocab, hidden], resident.
bool glm5_lm_head_init(const void* w, int64_t rows, int64_t cols);
bool glm5_lm_head(const float* x, float* logits, int64_t rows, int64_t cols);

// grouped per-head projection: x is [groups, cols] contiguous, y is [groups, rows].
bool glm5_q8_proj_grouped(const void* w, const float* x, float* y,
                          int64_t rows, int64_t cols, int groups);

void glm5_gpu_stats();

// Register a model shard so expert fills read via ReadFile (Windows) instead
// of page-faulting the mmap. base = mmap pointer of the shard's data section,
// file_data_offset = that section's offset inside the file.
void glm5_register_shard(const void* base, size_t length, const char* path, uint64_t file_data_offset);

// Demand-load a union of experts into VRAM slots (chunked prefill).
void glm5_gpu_load_experts(const void* gate_base, const void* up_base, const void* down_base,
                           int64_t ff, const int* experts, int n);

// Fast-path MoE staging: fills VRAM slots for the selected experts and
// writes the device pointer table [gate[8], up[8], down[8]] (24 uint64
// device addresses) via async H2D on the user stream. Weights on device
// (written by the router kernel) are NOT touched here.
bool glm5_gpu_stage_experts(const void* gate_base, const void* up_base, const void* down_base,
                            int64_t ff, const int* experts, int n,
                            void* d_etable, void* user_stream);

// Layer-ahead prefetch into the pinned RAM ring (off the critical path).
// ids come from the proxy router; misses here cost nothing at use time
// except a normal fill.
void glm5_prefetch_layer(const void* gate_base, const void* up_base, const void* down_base,
                         int64_t slice, const int* experts, int n);

// OS page-cache warming for mmap'd expert slices (fadvise-WILLNEED style):
// asynchronous, zero buffers, wrong predictions only cost idle disk.
void glm5_warm_slices(const void* gate_base, const void* up_base, const void* down_base,
                      int64_t slice, const int* experts, int n);

} // namespace glm5
} // namespace nt
