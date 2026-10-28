#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace nt {
namespace ds41 {

// Expert-offload to a CUDA device, for MXFP4 quantized expert tensors.
// Initialization is lazy; disabled with DS41_GPU=0. VRAM budget: DS41_VRAM_GB (default 40).

bool gpu_experts_enabled();

// Fused routed-MoE computation:
//   gate/up: [n_expert, ff, hidden] MXFP4;  down: [n_expert, hidden, ff] MXFP4
//   out[hidden] = sum_k weights[k] * down[expert_k] @ swiglu_clamp(gate[expert_k] @ x, up[expert_k] @ x)
// Returns false when the GPU path is unavailable (caller falls back to the CPU path).
bool gpu_moe_experts(const float* x, int64_t hidden,
                     const void* gate_base, const void* up_base, const void* down_base,
                     int64_t ff,
                     const int* experts, const float* weights, int n_used,
                     float limit, float* out);

// Runs independent CPU work on the calling thread after queuing the GPU
// kernels/readback, before waiting for completion. The callback runs at most
// once and must not call DS41 GPU helpers or modify the input/weight buffers.
// It may have run even if the final GPU wait fails; callers track that state.
// Output is published only after the wait. All queued work is drained before
// return (including failure) or propagation of a callback exception.
bool gpu_moe_experts(const float* x, int64_t hidden,
                     const void* gate_base, const void* up_base, const void* down_base,
                     int64_t ff, const int* experts, const float* weights, int n_used,
                     float limit, float* out, const std::function<void()>& host_work);

// single-slice GEMV probe (for equivalence testing)
bool gpu_gemv_expert(const void* base, int64_t expert, int64_t rows, int64_t cols,
                     const float* x, float* y);

// LM head on device: upload once (BF16 rows x cols), then gemv per token.
bool gpu_lm_head_init(const void* weight, int64_t rows, int64_t cols);
bool gpu_lm_head(const float* x, float* logits, int64_t rows, int64_t cols);

// ------------------------------------------------------------
// Backbone (attention) offload: weights resident in VRAM.
// All tensors are Q8_0 unless noted.
// ------------------------------------------------------------
bool gpu_backbone_enabled();

// qr = rmsnorm(wq_a @ x) ; q = rope(wq_b @ qr) ; kv = rope(rmsnorm(wkv @ x))
// cos_t/sin_t: rope table (rope_dim/2 entries each), offset = head_dim - rope_dim
bool gpu_attn_prologue(const float* x, int64_t hidden,
                       const void* wq_a, const void* q_a_norm,
                       const void* wq_b,
                       const void* wkv, const void* kv_a_norm,
                       int64_t q_lora, int64_t head_dim, int64_t n_heads, int64_t rope_dim, float eps,
                       const float* cos_t, const float* sin_t,
                       float* qr_out, float* q_out, float* kv_out);

// attn_out [heads*head_dim] -> out [hidden]  (grouped wo_a + wo_b)
// Independent wo_a groups share one launch. DS41_GROUPED_TAIL=0 restores
// one launch per group; read once on the first tail invocation.
bool gpu_attn_tail(const float* attn_out, int64_t head_dim, int64_t n_heads,
                   const void* wo_a, const void* wo_b,
                   int64_t o_groups, int64_t o_lora, int64_t hidden,
                   float* out);

// Cumulative successful tail calls and their projection launches. Snapshot
// deltas distinguish GPU coverage and scheduling without timing every kernel.
struct BackboneStats {
    uint64_t tail_calls = 0;
    uint64_t grouped_tail_calls = 0;
    uint64_t tail_gemv_launches = 0;
};
BackboneStats gpu_backbone_stats();

// compressor state: skv = comp_wkv @ x (BF16); sscore = comp_gate @ x (BF16, optional)
bool gpu_comp_state(const float* x, int64_t hidden, int64_t latent,
                    const void* comp_wkv, const void* comp_gate,
                    float* skv_out, float* sscore_out);

// indexer: idxq = idx_q_b @ qr (Q8_0); idxw = idx_proj @ x (BF16); idxk = idx_k @ latent (BF16)
bool gpu_indexer_gemv(const void* w, const float* x, int64_t hidden_in, int64_t out_rows,
                      int dtypesel, float* out);

// uploads and keeps a tensor resident; returns nullptr on failure
const void* gpu_resident(const void* host_data, size_t bytes);

void gpu_print_stats();

} // namespace ds41
} // namespace nt
