#pragma once

#include "../core/types.h"
#include "../model/config.h"
#include "../model/loader.h"
#include <vector>
#include <string>

namespace nt {

// ============================================================
// Layer weight pointers into the GPU buffer for a given slot
// ============================================================
struct LayerWeightPtrs {
    const void* attn_q;     DType attn_q_dtype;
    const void* attn_k;     DType attn_k_dtype;
    const void* attn_v;     DType attn_v_dtype;
    const void* attn_output; DType attn_o_dtype;
    const void* ffn_gate;   DType ffn_gate_dtype;
    const void* ffn_up;     DType ffn_up_dtype;
    const void* ffn_down;   DType ffn_down_dtype;
};

// ============================================================
// LayerStreamer: Double-buffer layer streaming via PCIe
//
// Two GPU buffers (slots 0 and 1). While layer N computes on
// slot A, layer N+1 transfers into slot B via async H2D copy.
// Uses CUDA events for synchronization.
// ============================================================

class LayerStreamer {
public:
    LayerStreamer() = default;
    ~LayerStreamer();

    // Non-copyable
    LayerStreamer(const LayerStreamer&) = delete;
    LayerStreamer& operator=(const LayerStreamer&) = delete;

    // Initialize: parse layer tensor info, allocate GPU buffers
    void init(const GGUFLoader& loader, const ModelConfig& config);

    // Free GPU buffers and events
    void shutdown();

    // Begin async H2D transfer of layer_idx into GPU buffer slot
    // Uses STREAM_TRANSFER0 for slot 0, STREAM_TRANSFER1 for slot 1
    void begin_transfer(int layer_idx, int slot);

    // Make compute stream wait until transfer into slot is done
    void wait_transfer(int slot);

    // Record that compute on slot is done (safe to overwrite)
    void signal_compute_done(int slot);

    // Get weight pointers for the given slot
    LayerWeightPtrs get_weights(int slot) const;

    // Total size of one GPU layer buffer
    size_t buffer_size() const { return buf_size_; }

private:
    // Per-tensor info within a layer buffer
    struct TensorSlot {
        size_t gpu_offset;      // offset within GPU buffer
        const void* cpu_ptr;    // mmap'd source (or pinned staging)
        size_t nbytes;          // transfer size
        DType dtype;
    };

    // Layout of all 7 tensors for one layer
    struct LayerLayout {
        TensorSlot attn_q, attn_k, attn_v, attn_output;
        TensorSlot ffn_gate, ffn_up, ffn_down;
    };

    void* gpu_buf_[2] = {};          // two GPU buffer slots
    size_t buf_size_ = 0;            // size of each buffer
    int current_layer_[2] = {-1, -1}; // which layer is in each slot

    std::vector<LayerLayout> layers_; // per-layer CPU→GPU mapping

    void* transfer_done_[2] = {};    // CUDA events: transfer finished
    void* compute_done_[2] = {};     // CUDA events: compute finished

    // Pinned memory strategy
    bool mmap_pinned_ = false;       // true if cudaHostRegister succeeded
    void* pinned_staging_ = nullptr; // fallback pinned buffer if register fails
    size_t pinned_size_ = 0;

    // Helper to compute byte size of a weight tensor
    static size_t tensor_bytes(const GGUFTensorInfo& info);
};

} // namespace nt
