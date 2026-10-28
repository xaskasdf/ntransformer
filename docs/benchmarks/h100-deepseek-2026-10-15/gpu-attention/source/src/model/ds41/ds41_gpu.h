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
// Independent experts share four kernel launches; DS41_MOE_BATCH=0 restores
// five launches per expert. Read once on first successful initialization.
// DS41_STAGING_POOL=1 and DS41_MXFP4_REGISTER_LUT=1 enable experimental
// scheduling/lookup controls; they did not improve the measured H100 workload.
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

// Q8 shared expert, with CPU AVX2-compatible dot-product reduction order.
// Weights must remain immutable and alive while the resident registry is used.
// The exponential is evaluated in FP64 then rounded to FP32; differences
// from the CPU expf approximation can still remain.
struct SharedExpert {
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    int64_t ff = 0;
    float limit = 0;
};
bool gpu_shared_expert(const float* x, int64_t hidden, const SharedExpert& shared, float* out);
// Both outputs are published only after successful stream completion. Shared
// and routed work share the device input and one completion wait; the caller
// performs the original routed-plus-shared addition.
bool gpu_moe_experts_shared(const float* x, int64_t hidden,
                            const void* gate, const void* up, const void* down, int64_t ff,
                            const int* experts, const float* weights, int n_used,
                            float limit, float* out, const SharedExpert& shared, float* shared_out);

// Cumulative successful calls; tracing is opt-in (DS41_MOE_TRACE=1, read once).
// Host phases: input enqueue, lookup, staging, weight-copy enqueue, kernel and
// readback enqueue, callback, completion wait, output publication.
// Stream phases: input/zero, preparation gap, weight uploads, compute/launch
// span, readback. CUDA-event spans include stream idle gaps, not just GPU busy
// time. Host and stream durations overlap and must not be added together.
// With DS41_SHARED_EARLY (default), shared compute is in the preparation span.
struct MoeGpuStats {
    uint64_t calls = 0;
    uint64_t batched_calls = 0;
    uint64_t kernel_launches = 0;
    uint64_t uploaded_bytes = 0;
    uint64_t traced_calls = 0;
    uint64_t trace_failures = 0;
    uint64_t shared_calls = 0;
    uint64_t shared_kernel_launches = 0;
    uint64_t shared_weight_bytes = 0;
    uint64_t staged_slices = 0;
    uint64_t staging_pool_calls = 0;
    uint64_t transient_copy_threads = 0;
    uint64_t shared_early_calls = 0;
    uint64_t register_lut_launches = 0;
    double host_seconds[8] = {};
    double stream_seconds[5] = {};
};
MoeGpuStats gpu_moe_stats();

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

// Session-owned raw KV ring and append-only compressed rows. Calls use the
// engine's single CUDA stream; they are not a concurrent execution interface.
struct GpuAttentionState;
GpuAttentionState* gpu_attention_create(int layers, int window, int heads, int dim);
void gpu_attention_destroy(GpuAttentionState* state);
void gpu_attention_reset(GpuAttentionState* state);
bool gpu_attention_ready(const GpuAttentionState* state, int layer, int64_t position);
struct AttentionRows {
    int source = -1;
    const float* rows = nullptr; // Immutable prefix; reset before replacing it.
    int count = 0;
    const int* selected = nullptr;
    int n_selected = 0;
};
bool gpu_attention_read_query(GpuAttentionState* state, float* query);
bool gpu_attention_tail(GpuAttentionState* state, int layer, int64_t position,
                        const AttentionRows& rows, const float* sinks,
                        const void* wo_a, const void* wo_b, int64_t o_groups,
                        int64_t o_lora, int64_t hidden, float* out);
// Independent numerical/state probe using host Q/KV instead of Q8 projections.
bool gpu_attention_probe(GpuAttentionState* state, int layer, int64_t position,
                         const float* query, const float* kv, const float* cos_t,
                         const float* sin_t, int rope_dim, const AttentionRows& rows,
                         const float* sinks, float* out);

// qr = rmsnorm(wq_a @ x) ; q = rope(wq_b @ qr) ; kv = rope(rmsnorm(wkv @ x))
// cos_t/sin_t: rope table (rope_dim/2 entries each), offset = head_dim - rope_dim
// DS41_QKV_READBACK_BATCH=0 restores three waits instead of one (read once).
// With resident state, Q stays on device and q_out is unused (may be null).
// qr_out and kv_out remain required host mirrors. Publish state only on success;
// callers recover Q with gpu_attention_read_query if the later tail falls back.
bool gpu_attn_prologue(const float* x, int64_t hidden,
                       const void* wq_a, const void* q_a_norm,
                       const void* wq_b,
                       const void* wkv, const void* kv_a_norm,
                       int64_t q_lora, int64_t head_dim, int64_t n_heads, int64_t rope_dim, float eps,
                       const float* cos_t, const float* sin_t,
                       float* qr_out, float* q_out, float* kv_out,
                       GpuAttentionState* resident = nullptr, int layer = 0, int64_t position = 0);

// attn_out [heads*head_dim] -> out [hidden]  (grouped wo_a + wo_b)
// Independent wo_a groups share one launch. DS41_GROUPED_TAIL=0 restores
// one launch per group; read once on the first tail invocation.
bool gpu_attn_tail(const float* attn_out, int64_t head_dim, int64_t n_heads,
                   const void* wo_a, const void* wo_b,
                   int64_t o_groups, int64_t o_lora, int64_t hidden,
                   float* out);

// Cumulative successful prologue/tail calls, waits and projection launches. Snapshot
// deltas distinguish GPU coverage and scheduling without timing every kernel.
struct BackboneStats {
    uint64_t tail_calls = 0;
    uint64_t grouped_tail_calls = 0;
    uint64_t tail_gemv_launches = 0;
    uint64_t prologue_calls = 0;
    uint64_t prologue_readback_waits = 0;
    uint64_t router_calls = 0;
    uint64_t hc_projection_calls = 0;
    uint64_t resident_attention_calls = 0;
    uint64_t resident_comp_upload_bytes = 0;
    uint64_t resident_metadata_upload_bytes = 0;
    uint64_t resident_attention_launches = 0;
    uint64_t prologue_readback_bytes = 0;
    uint64_t tail_upload_bytes = 0;
};
BackboneStats gpu_backbone_stats();

// Resident F32/BF16 router weights; output is published after successful readback.
// avx2_order must match the CPU matvec build. Weights stay immutable and alive.
// Session uses this by default; DS41_ROUTER_GPU=0 restores its CPU projection.
bool gpu_router(const void* w, const float* x, float* out, int64_t rows, int64_t cols,
                 bool bf16, bool avx2_order);

// Same arithmetic and ownership contract for the small HC coefficient matrices.
bool gpu_hc_projection(const void* w, const float* x, float* out, int64_t rows, int64_t cols,
                        bool bf16, bool avx2_order);

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
