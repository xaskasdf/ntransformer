#pragma once

#include "../core/types.h"
#include "../model/config.h"
#include "../model/loader.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/sysinfo.h>

#ifdef USE_GPUNVME
#include <gpunvme/layer_loader.h>
#endif

namespace nt {

// ============================================================
// 3-Tier Adaptive Layer Caching
// ============================================================
enum class LayerTier { VRAM, RAM, NVME };

struct TierConfig {
    int n_vram = 0;              // tier A count (GPU resident)
    int n_ram  = 0;              // tier B count (pinned host RAM)
    int n_nvme = 0;              // tier C count (NVMe cold storage)
    size_t vram_used = 0;        // bytes allocated for tier A
    size_t ram_used  = 0;        // bytes allocated for tier B
    size_t layer_bytes = 0;      // per-layer buffer size

    // Auto-compute tier sizes from available hardware resources
    static TierConfig compute(int n_layers, size_t layer_bytes,
                              size_t vram_reserve = 512ULL << 20,
                              size_t ram_reserve = 6ULL << 30);
    void print() const;
};

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

    // Initialize with 3-tier adaptive caching (VRAM + RAM + NVMe)
    void init_tiered(const GGUFLoader& loader, const ModelConfig& config);

    // Free GPU buffers and events
    void shutdown();

    // Begin async H2D transfer of layer_idx into GPU buffer slot
    // Uses STREAM_TRANSFER0 for slot 0, STREAM_TRANSFER1 for slot 1
    void begin_transfer(int layer_idx, int slot);

    // Queue a background CPU memcpy from mmap to staging[slot] (non-blocking)
    // No-op if mmap is pinned (direct DMA path).
    void prefetch_staging(int layer_idx, int slot);

    // Wait for staging[slot] to be filled, then issue async H2D to gpu[slot]
    void begin_h2d(int layer_idx, int slot);

    // Make compute stream wait until transfer into slot is done
    void wait_transfer(int slot);

    // Record that compute on slot is done (safe to overwrite)
    void signal_compute_done(int slot);

    // Get weight pointers for the given slot
    LayerWeightPtrs get_weights(int slot) const;

    // Get weight pointers for a VRAM-resident layer (tier A)
    LayerWeightPtrs get_resident_weights(int layer_idx) const;

    // Get weight pointers for a RAM-cached layer (tier B) — zero-copy path
    // Returns pointers into pinned host memory (GPU-accessible via PCIe)
    LayerWeightPtrs get_ram_weights(int layer_idx) const;

    // Tier queries
    bool is_vram_resident(int layer_idx) const;
    LayerTier layer_tier(int layer_idx) const;
    const TierConfig& tier_config() const { return tier_config_; }

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

    // Double staging buffers (used when mmap is NOT pinned)
    void* staging_buf_[2] = {};      // two pinned staging buffers
    size_t staging_size_ = 0;

    // Worker thread for background CPU memcpy (mmap → staging)
    std::thread worker_thread_;
    std::mutex  worker_mutex_;
    std::condition_variable worker_cv_;
    std::condition_variable staging_ready_cv_;
    bool staging_ready_[2] = {};     // staging[slot] has been filled
    bool worker_shutdown_ = false;

    struct WorkerRequest { int layer_idx; int slot; };
    WorkerRequest worker_request_ = {};
    bool worker_has_work_ = false;

    void worker_loop();              // worker thread entry point
    void memcpy_layer_to_staging(int layer_idx, int slot);

    // 3-tier adaptive caching state
    TierConfig tier_config_;
    std::vector<LayerTier> layer_tier_;      // per-layer assignment [n_layers]
    std::vector<void*> vram_resident_;       // VRAM buffers [n_vram]
    std::vector<void*> ram_cache_;           // pinned RAM buffers [n_ram]
    bool tiered_mode_ = false;

    // Helper: compute total transfer size for a layer
    size_t layer_transfer_size(int layer_idx) const;

    // Helper to compute byte size of a weight tensor
    static size_t tensor_bytes(const GGUFTensorInfo& info);

#ifdef USE_GPUNVME
    // gpu-nvme-direct Layer Loader
    gpunvme_layer_loader_t nvme_loader_ = {};
    bool nvme_initialized_ = false;

    struct NvmeLayerInfo {
        uint64_t start_lba;     // LBA of first byte of this layer's tensor data
        size_t   total_bytes;   // total bytes for all 7 tensors
    };
    std::vector<NvmeLayerInfo> nvme_layers_;
    uint64_t gguf_start_lba_ = 0;
    uint32_t nvme_block_size_ = 512;
#endif
};

} // namespace nt
