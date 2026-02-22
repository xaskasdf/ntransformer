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
// Delta weight pointers: 7 U/V pairs for delta-encoded streaming
// ============================================================
struct DeltaWeightPtrs {
    const void* attn_q_U;  const void* attn_q_V;
    const void* attn_k_U;  const void* attn_k_V;
    const void* attn_v_U;  const void* attn_v_V;
    const void* attn_o_U;  const void* attn_o_V;
    const void* ffn_gate_U; const void* ffn_gate_V;
    const void* ffn_up_U;   const void* ffn_up_V;
    const void* ffn_down_U; const void* ffn_down_V;
};

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

    // Enable Q6_K → Q4_K_M requantization for tier B (call before init_tiered)
    void set_requant_q4k(bool enable) { requant_tier_b_ = enable; }

    // Total size of one GPU layer buffer
    size_t buffer_size() const { return buf_size_; }

    // Delta encoding support
    bool init_delta(const std::string& ntd_path, const ModelConfig& config);
    bool is_delta_mode() const { return delta_mode_; }
    int delta_rank() const { return delta_rank_; }

    // Get base weight pointer for weight index (0-6: q,k,v,o,gate,up,down)
    const void* base_weight_ptr(int weight_idx) const;

    // Get delta U/V pointers from the current GPU buffer slot
    DeltaWeightPtrs get_delta_weights(int slot) const;

    // VRAM buffer for temporary rank-sized vector (rank floats)
    float* delta_temp_buf() const { return delta_temp_; }

private:
    // Per-tensor info within a layer buffer
    struct TensorSlot {
        size_t gpu_offset;      // offset within GPU buffer
        const void* cpu_ptr;    // mmap'd source (or pinned staging)
        size_t nbytes;          // transfer size (original)
        size_t xfer_nbytes;     // actual H2D transfer size (may be smaller if requantized)
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
    bool requant_tier_b_ = false;            // Q6_K → Q4_K_M for tier B

    // Q6_K → Q4_K_M in-place requantization (returns new byte count)
    static size_t requantize_q6k_to_q4km(void* data, size_t nbytes_q6k);

    // Helper: compute total transfer size for a layer
    size_t layer_transfer_size(int layer_idx) const;

    // Helper to compute byte size of a weight tensor
    static size_t tensor_bytes(const GGUFTensorInfo& info);

    // Delta encoding state
    bool delta_mode_ = false;
    int delta_rank_ = 0;
    void* delta_base_buf_ = nullptr;        // VRAM: 7 Q6_K base matrices (~669 MB)
    size_t delta_base_size_ = 0;
    float* delta_temp_ = nullptr;           // VRAM: rank floats scratch
    void* delta_mmap_ = nullptr;            // mmap'd .ntd file
    size_t delta_mmap_size_ = 0;
    int delta_mmap_fd_ = -1;

    // Per-weight-type base offsets within delta_base_buf_ (7 entries)
    size_t delta_base_offsets_[7] = {};
    DType delta_base_dtypes_[7] = {};

    // Per-layer delta layout: offset within .ntd file for each U/V pair
    struct DeltaTensorInfo {
        size_t file_offset;     // offset from .ntd start
        size_t nbytes;          // F16 tensor bytes
    };
    struct DeltaLayerLayout {
        DeltaTensorInfo U[7];   // U matrices for 7 weight types
        DeltaTensorInfo V[7];   // V matrices
    };
    std::vector<DeltaLayerLayout> delta_layers_;

    // Per-layer delta buffer layout: offset within gpu_buf_ slot
    struct DeltaSlotLayout {
        size_t U_offset[7];
        size_t V_offset[7];
    };
    DeltaSlotLayout delta_slot_layout_ = {};
    size_t delta_buf_size_ = 0;             // per-slot delta buffer size

#ifdef USE_GPUNVME
    // gpu-nvme-direct Layer Loader
    gpunvme_layer_loader_t nvme_loader_ = {};
    bool nvme_initialized_ = false;

    struct NvmeTensorMap {
        size_t read_offset;     // offset within NVMe read buffer
        size_t gpu_offset;      // offset within staging/GPU buffer
        size_t nbytes;
        uint64_t start_lba;     // per-tensor start LBA (for BAR1 scatter reads)
        size_t   lba_aligned_bytes; // LBA-aligned read size for this tensor
        size_t   lba_sub_offset;    // sub-LBA offset within first block
    };
    struct NvmeLayerInfo {
        uint64_t start_lba;     // first LBA of this layer's file span
        size_t   read_bytes;    // NVMe read size (LBA-aligned)
        NvmeTensorMap tensors[7];
    };
    std::vector<NvmeLayerInfo> nvme_layers_;
    uint64_t gguf_start_lba_ = 0;
    uint32_t nvme_block_size_ = 512;
    void* nvme_read_buf_ = nullptr;
    size_t nvme_read_buf_size_ = 0;

    // BAR1 direct VRAM mode (Tier 2)
    bool bar1_enabled_ = false;
    uint64_t bar1_phys_[2] = {};     // BAR1 physical addr of gpu_buf_[0], gpu_buf_[1]

    // BAR1 bulk read: single VRAM temp buffer for entire layer span
    void* nvme_vram_temp_ = nullptr;         // VRAM temp for BAR1 bulk reads
    uint64_t nvme_vram_temp_bar1_phys_ = 0;  // BAR1 physical address of temp
    size_t nvme_vram_temp_size_ = 0;
#endif
};

} // namespace nt
