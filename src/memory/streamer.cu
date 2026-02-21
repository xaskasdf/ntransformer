#include "streamer.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sys/sysinfo.h>

namespace nt {

// ============================================================
// TierConfig: auto-compute tier sizes from hardware
// ============================================================
TierConfig TierConfig::compute(int n_layers, size_t layer_bytes,
                               size_t vram_reserve, size_t ram_reserve) {
    TierConfig tc;
    tc.layer_bytes = layer_bytes;

    // Query available VRAM
    size_t vram_free = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free, &vram_total);

    // Query available RAM via /proc/meminfo (includes reclaimable page cache)
    size_t ram_free = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            size_t val;
            if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) {
                ram_free = val * 1024;
                break;
            }
        }
        fclose(f);
    }
    if (ram_free == 0) {
        // Fallback to sysinfo if /proc/meminfo unavailable
        struct sysinfo si;
        sysinfo(&si);
        ram_free = (size_t)si.freeram * si.mem_unit +
                   (size_t)si.bufferram * si.mem_unit;
    }

    fprintf(stderr, "TierConfig: VRAM free=%.1f GB, RAM available=%.1f GB\n",
            vram_free / (1024.0 * 1024 * 1024), ram_free / (1024.0 * 1024 * 1024));
    fprintf(stderr, "TierConfig: VRAM reserve=%.1f GB, RAM reserve=%.1f GB\n",
            vram_reserve / (1024.0 * 1024 * 1024), ram_reserve / (1024.0 * 1024 * 1024));

    size_t vram_avail = (vram_free > vram_reserve) ? (vram_free - vram_reserve) : 0;
    size_t ram_avail  = (ram_free > ram_reserve) ? (ram_free - ram_reserve) : 0;

    tc.n_vram = std::min(n_layers, (int)(vram_avail / layer_bytes));
    int remaining = n_layers - tc.n_vram;
    tc.n_ram  = std::min(remaining, (int)(ram_avail / layer_bytes));
    tc.n_nvme = n_layers - tc.n_vram - tc.n_ram;

    tc.vram_used = (size_t)tc.n_vram * layer_bytes;
    tc.ram_used  = (size_t)tc.n_ram * layer_bytes;

    return tc;
}

void TierConfig::print() const {
    fprintf(stderr, "TierConfig: %d VRAM (%.1f GB) + %d RAM (%.1f GB) + %d NVMe\n",
            n_vram, vram_used / (1024.0 * 1024 * 1024),
            n_ram,  ram_used / (1024.0 * 1024 * 1024),
            n_nvme);
    fprintf(stderr, "  Per-layer: %.1f MB\n", layer_bytes / (1024.0 * 1024));
}

// ============================================================
// Helper: compute byte size from GGUF tensor info
// ============================================================
size_t LayerStreamer::tensor_bytes(const GGUFTensorInfo& info) {
    return info.nbytes;
}

// ============================================================
// Destructor
// ============================================================
LayerStreamer::~LayerStreamer() {
    shutdown();
}

// ============================================================
// Init: build per-layer layout, allocate GPU double-buffers
// ============================================================
void LayerStreamer::init(const GGUFLoader& loader, const ModelConfig& config) {
    int n_layers = config.n_layers;
    layers_.resize(n_layers);

    const char* tensor_suffixes[] = {
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"
    };

    size_t max_layer_bytes = 0;

    for (int i = 0; i < n_layers; i++) {
        std::string pfx = "blk." + std::to_string(i) + ".";
        LayerLayout& lay = layers_[i];

        TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };

        size_t offset = 0;

        for (int t = 0; t < 7; t++) {
            std::string name = pfx + tensor_suffixes[t];
            const GGUFTensorInfo* info = loader.tensor_info(name);
            NT_CHECK(info != nullptr, ("Missing tensor: " + name).c_str());

            // Align offset to 256 bytes for efficient GPU access
            offset = (offset + 255) & ~(size_t)255;

            slots[t]->gpu_offset = offset;
            slots[t]->cpu_ptr    = loader.tensor_data(name);
            slots[t]->nbytes     = info->nbytes;
            slots[t]->dtype      = ggml_to_dtype(info->ggml_type);

            NT_CHECK(slots[t]->cpu_ptr != nullptr, ("Null data for tensor: " + name).c_str());

            offset += info->nbytes;
        }

        size_t layer_bytes = (offset + 255) & ~(size_t)255;
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }

    buf_size_ = max_layer_bytes;

    fprintf(stderr, "LayerStreamer: %d layers, buffer size: %.1f MB each (%.1f MB total for 2 buffers)\n",
        n_layers, buf_size_ / (1024.0 * 1024.0), 2.0 * buf_size_ / (1024.0 * 1024.0));

    // Allocate two GPU buffers
    for (int s = 0; s < 2; s++) {
        cudaError_t err = cudaMalloc(&gpu_buf_[s], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate GPU layer buffer");
        current_layer_[s] = -1;
    }

    // Create CUDA events (disable timing for lower overhead)
    for (int s = 0; s < 2; s++) {
        cudaEvent_t ev;
        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        transfer_done_[s] = static_cast<void*>(ev);

        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        compute_done_[s] = static_cast<void*>(ev);
    }

    // Try to pin the mmap'd region for true async DMA
    const void* data_ptr = loader.mmap_data_ptr();
    size_t data_size = loader.tensor_data_size();

    cudaError_t pin_err = cudaHostRegister(
        const_cast<void*>(data_ptr), data_size, cudaHostRegisterReadOnly);

    if (pin_err == cudaSuccess) {
        mmap_pinned_ = true;
        fprintf(stderr, "LayerStreamer: mmap region pinned (%.1f GB) — true async DMA enabled\n",
            data_size / (1024.0 * 1024.0 * 1024.0));
    } else {
        mmap_pinned_ = false;
        fprintf(stderr, "LayerStreamer: cudaHostRegister failed (%s), using double pinned staging buffers\n",
            cudaGetErrorString(pin_err));

        // Allocate TWO pinned staging buffers for pipelined overlap
        staging_size_ = buf_size_;
        for (int s = 0; s < 2; s++) {
            cudaError_t err = cudaMallocHost(&staging_buf_[s], staging_size_);
            NT_CHECK(err == cudaSuccess, "Failed to allocate pinned staging buffer");
        }
        fprintf(stderr, "LayerStreamer: double pinned staging: %.1f MB x 2 = %.1f MB\n",
            staging_size_ / (1024.0 * 1024.0), 2.0 * staging_size_ / (1024.0 * 1024.0));

        // Start worker thread for background CPU memcpy
        worker_shutdown_ = false;
        worker_has_work_ = false;
        staging_ready_[0] = false;
        staging_ready_[1] = false;
        worker_thread_ = std::thread(&LayerStreamer::worker_loop, this);
        fprintf(stderr, "LayerStreamer: worker thread started\n");
    }

    // Record initial compute_done events so the first begin_transfer doesn't deadlock
    auto& dev = CUDADevice::instance();
    for (int s = 0; s < 2; s++) {
        dev.record_event(compute_done_[s], STREAM_COMPUTE);
    }

#ifdef USE_GPUNVME
    // Try to initialize gpu-nvme-direct Layer Loader
    const char* nvme_bdf = getenv("GPUNVME_PCI_BDF");
    const char* nvme_lba_str = getenv("GPUNVME_GGUF_LBA");

    if (nvme_bdf && nvme_lba_str) {
        uint64_t gguf_start_lba = strtoull(nvme_lba_str, NULL, 0);

        gpunvme_err_t err = gpunvme_layer_loader_init(
            &nvme_loader_, nvme_bdf, buf_size_, /*pipeline_depth=*/32);

        if (err == GPUNVME_OK) {
            nvme_initialized_ = true;
            gguf_start_lba_ = gguf_start_lba;
            nvme_block_size_ = gpunvme_layer_loader_block_size(&nvme_loader_);

            // Pre-compute per-layer LBAs
            nvme_layers_.resize(n_layers);
            for (int i = 0; i < n_layers; i++) {
                std::string first = "blk." + std::to_string(i) + ".attn_q.weight";
                uint64_t byte_offset = loader.tensor_file_offset(first);
                nvme_layers_[i].start_lba = gguf_start_lba + (byte_offset / nvme_block_size_);
                nvme_layers_[i].total_bytes = layer_transfer_size(i);
            }

            fprintf(stderr, "LayerStreamer: NVMe backend OK (MDTS=%uK, block=%u)\n",
                    gpunvme_layer_loader_max_transfer(&nvme_loader_) / 1024,
                    nvme_block_size_);
        } else {
            fprintf(stderr, "LayerStreamer: NVMe init failed (err=%d), fallback to mmap\n", err);
        }
    }
#endif

    fprintf(stderr, "LayerStreamer: initialized\n");
}

// ============================================================
// Init tiered: allocate VRAM + RAM caches, then init pipeline
// ============================================================
void LayerStreamer::init_tiered(const GGUFLoader& loader, const ModelConfig& config) {
    // Step 1: call init() for layout parsing + double-buffer setup
    init(loader, config);

    int n_layers = config.n_layers;

    // Step 2: compute VRAM reserve for inference buffers that will be allocated later
    // KV cache: 2 (K+V) × n_layers × max_seq × n_kv_heads × head_dim × sizeof(float)
    size_t kv_bytes = 2ULL * n_layers * config.max_seq_len * config.n_kv_heads
                      * config.head_dim * sizeof(float);
    // Workspace: max(attn, ffn) — FFN typically dominates
    size_t attn_ws = (size_t)config.max_seq_len * (config.n_heads + 2 * config.n_kv_heads
                     + config.n_heads) * config.head_dim * sizeof(float);
    size_t ffn_ws  = 2ULL * config.max_seq_len * config.intermediate_size * sizeof(float);
    size_t workspace_bytes = std::max(attn_ws, ffn_ws);
    // Hidden + residual + logits + positions
    size_t misc_bytes = 2ULL * config.max_seq_len * config.hidden_size * sizeof(float)
                        + config.vocab_size * sizeof(float)
                        + config.max_seq_len * sizeof(int);
    size_t vram_reserve = kv_bytes + workspace_bytes + misc_bytes + (256ULL << 20);

    fprintf(stderr, "LayerStreamer: VRAM reserve for inference: %.1f GB "
            "(KV=%.1f, WS=%.1f, misc=%.1f)\n",
            vram_reserve / (1024.0 * 1024 * 1024),
            kv_bytes / (1024.0 * 1024 * 1024),
            workspace_bytes / (1024.0 * 1024 * 1024),
            misc_bytes / (1024.0 * 1024 * 1024));

    // Step 3: compute tier sizes
    tier_config_ = TierConfig::compute(n_layers, buf_size_, vram_reserve);
    tier_config_.print();

    // Step 3: assign per-layer tiers
    layer_tier_.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        if (i < tier_config_.n_vram)
            layer_tier_[i] = LayerTier::VRAM;
        else if (i < tier_config_.n_vram + tier_config_.n_ram)
            layer_tier_[i] = LayerTier::RAM;
        else
            layer_tier_[i] = LayerTier::NVME;
    }

    // Step 4: Tier A — allocate VRAM buffers and load from mmap
    vram_resident_.resize(tier_config_.n_vram);
    for (int i = 0; i < tier_config_.n_vram; i++) {
        cudaError_t err = cudaMalloc(&vram_resident_[i], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate VRAM resident buffer");

        // Copy all 7 tensors from mmap to VRAM
        const LayerLayout& lay = layers_[i];
        const TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };
        uint8_t* gpu_base = static_cast<uint8_t*>(vram_resident_[i]);
        for (int t = 0; t < 7; t++) {
            nt_cuda_memcpy_h2d(gpu_base + slots[t]->gpu_offset,
                               slots[t]->cpu_ptr, slots[t]->nbytes);
        }

        if ((i + 1) % 10 == 0 || i == tier_config_.n_vram - 1) {
            fprintf(stderr, "  Loaded tier A layer %d/%d to VRAM\n", i + 1, tier_config_.n_vram);
        }
    }

    // Step 5: Tier B — allocate pinned RAM buffers and copy from mmap
    ram_cache_.resize(tier_config_.n_ram);
    for (int i = 0; i < tier_config_.n_ram; i++) {
        int layer_idx = tier_config_.n_vram + i;
        cudaError_t err = cudaMallocHost(&ram_cache_[i], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate pinned RAM cache buffer");

        // Copy all 7 tensors from mmap to pinned RAM
        const LayerLayout& lay = layers_[layer_idx];
        const TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };
        uint8_t* ram_base = static_cast<uint8_t*>(ram_cache_[i]);
        for (int t = 0; t < 7; t++) {
            memcpy(ram_base + slots[t]->gpu_offset,
                   slots[t]->cpu_ptr, slots[t]->nbytes);
        }

        if ((i + 1) % 10 == 0 || i == tier_config_.n_ram - 1) {
            fprintf(stderr, "  Loaded tier B layer %d/%d to pinned RAM\n",
                    i + 1, tier_config_.n_ram);
        }
    }

    tiered_mode_ = true;

    fprintf(stderr, "LayerStreamer: tiered init complete\n");
    fprintf(stderr, "  Free VRAM: %.1f GB\n",
            CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));
}

// ============================================================
// Shutdown: free everything
// ============================================================
void LayerStreamer::shutdown() {
    // Free tier caches
    for (auto* p : vram_resident_) {
        if (p) cudaFree(p);
    }
    vram_resident_.clear();
    for (auto* p : ram_cache_) {
        if (p) cudaFreeHost(p);
    }
    ram_cache_.clear();
    layer_tier_.clear();
    tiered_mode_ = false;

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        gpunvme_layer_loader_destroy(&nvme_loader_);
        nvme_initialized_ = false;
        fprintf(stderr, "LayerStreamer: NVMe backend shut down\n");
    }
#endif

    // Shut down worker thread first
    if (worker_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            worker_shutdown_ = true;
        }
        worker_cv_.notify_one();
        worker_thread_.join();
    }

    for (int s = 0; s < 2; s++) {
        if (gpu_buf_[s]) { cudaFree(gpu_buf_[s]); gpu_buf_[s] = nullptr; }
        if (transfer_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(transfer_done_[s]));
            transfer_done_[s] = nullptr;
        }
        if (compute_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(compute_done_[s]));
            compute_done_[s] = nullptr;
        }
    }

    mmap_pinned_ = false;

    for (int s = 0; s < 2; s++) {
        if (staging_buf_[s]) {
            cudaFreeHost(staging_buf_[s]);
            staging_buf_[s] = nullptr;
        }
    }
    staging_size_ = 0;

    layers_.clear();
    buf_size_ = 0;
}

// ============================================================
// Begin async transfer of layer_idx into GPU buffer slot
// ============================================================
void LayerStreamer::begin_transfer(int layer_idx, int slot) {
    // Legacy API: synchronous staging + H2D. Use prefetch_staging + begin_h2d
    // for the pipelined path.
    prefetch_staging(layer_idx, slot);
    begin_h2d(layer_idx, slot);
}

// ============================================================
// Wait for transfer into slot to complete (called from compute stream)
// ============================================================
void LayerStreamer::wait_transfer(int slot) {
    auto& dev = CUDADevice::instance();
    dev.wait_event(STREAM_COMPUTE, transfer_done_[slot]);
}

// ============================================================
// Signal that compute on slot is done (called from compute stream)
// ============================================================
void LayerStreamer::signal_compute_done(int slot) {
    auto& dev = CUDADevice::instance();
    dev.record_event(compute_done_[slot], STREAM_COMPUTE);
}

// ============================================================
// Get weight pointers for a given slot
// Uses the layer that was most recently transferred to this slot.
// ============================================================
LayerWeightPtrs LayerStreamer::get_weights(int slot) const {
    NT_CHECK(slot == 0 || slot == 1, "Slot must be 0 or 1");
    NT_CHECK(current_layer_[slot] >= 0, "No layer transferred to this slot yet");

    const LayerLayout& lay = layers_[current_layer_[slot]];
    const uint8_t* gpu_base = static_cast<const uint8_t*>(gpu_buf_[slot]);

    LayerWeightPtrs wp;
    wp.attn_q       = gpu_base + lay.attn_q.gpu_offset;
    wp.attn_q_dtype = lay.attn_q.dtype;
    wp.attn_k       = gpu_base + lay.attn_k.gpu_offset;
    wp.attn_k_dtype = lay.attn_k.dtype;
    wp.attn_v       = gpu_base + lay.attn_v.gpu_offset;
    wp.attn_v_dtype = lay.attn_v.dtype;
    wp.attn_output  = gpu_base + lay.attn_output.gpu_offset;
    wp.attn_o_dtype = lay.attn_output.dtype;
    wp.ffn_gate       = gpu_base + lay.ffn_gate.gpu_offset;
    wp.ffn_gate_dtype = lay.ffn_gate.dtype;
    wp.ffn_up         = gpu_base + lay.ffn_up.gpu_offset;
    wp.ffn_up_dtype   = lay.ffn_up.dtype;
    wp.ffn_down       = gpu_base + lay.ffn_down.gpu_offset;
    wp.ffn_down_dtype = lay.ffn_down.dtype;

    return wp;
}

// ============================================================
// Get weight pointers for a VRAM-resident layer (tier A)
// ============================================================
LayerWeightPtrs LayerStreamer::get_resident_weights(int layer_idx) const {
    NT_CHECK(tiered_mode_ && layer_idx < tier_config_.n_vram,
             "get_resident_weights: layer not VRAM-resident");

    const LayerLayout& lay = layers_[layer_idx];
    const uint8_t* gpu_base = static_cast<const uint8_t*>(vram_resident_[layer_idx]);

    LayerWeightPtrs wp;
    wp.attn_q       = gpu_base + lay.attn_q.gpu_offset;
    wp.attn_q_dtype = lay.attn_q.dtype;
    wp.attn_k       = gpu_base + lay.attn_k.gpu_offset;
    wp.attn_k_dtype = lay.attn_k.dtype;
    wp.attn_v       = gpu_base + lay.attn_v.gpu_offset;
    wp.attn_v_dtype = lay.attn_v.dtype;
    wp.attn_output  = gpu_base + lay.attn_output.gpu_offset;
    wp.attn_o_dtype = lay.attn_output.dtype;
    wp.ffn_gate       = gpu_base + lay.ffn_gate.gpu_offset;
    wp.ffn_gate_dtype = lay.ffn_gate.dtype;
    wp.ffn_up         = gpu_base + lay.ffn_up.gpu_offset;
    wp.ffn_up_dtype   = lay.ffn_up.dtype;
    wp.ffn_down       = gpu_base + lay.ffn_down.gpu_offset;
    wp.ffn_down_dtype = lay.ffn_down.dtype;

    return wp;
}

// ============================================================
// Tier query helpers
// ============================================================
bool LayerStreamer::is_vram_resident(int layer_idx) const {
    return tiered_mode_ && layer_idx < tier_config_.n_vram;
}

LayerTier LayerStreamer::layer_tier(int layer_idx) const {
    if (!tiered_mode_) return LayerTier::NVME;
    return layer_tier_[layer_idx];
}

// ============================================================
// Helper: compute total contiguous transfer size for a layer
// ============================================================
size_t LayerStreamer::layer_transfer_size(int layer_idx) const {
    const LayerLayout& lay = layers_[layer_idx];
    const TensorSlot* slots[] = {
        &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
        &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
    };
    size_t total = 0;
    for (int t = 0; t < 7; t++) {
        size_t end = slots[t]->gpu_offset + slots[t]->nbytes;
        if (end > total) total = end;
    }
    return total;
}

// ============================================================
// Worker thread: background CPU memcpy from mmap to staging
// ============================================================
void LayerStreamer::worker_loop() {
    while (true) {
        int layer_idx, slot;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_cv_.wait(lock, [&] { return worker_has_work_ || worker_shutdown_; });
            if (worker_shutdown_) return;
            layer_idx = worker_request_.layer_idx;
            slot = worker_request_.slot;
            worker_has_work_ = false;
        }

        memcpy_layer_to_staging(layer_idx, slot);

        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
        }
        staging_ready_cv_.notify_all();
    }
}

// ============================================================
// Copy all 7 tensors from mmap to staging buffer[slot]
// ============================================================
void LayerStreamer::memcpy_layer_to_staging(int layer_idx, int slot) {
    const LayerLayout& lay = layers_[layer_idx];
    uint8_t* staging = static_cast<uint8_t*>(staging_buf_[slot]);

    const TensorSlot* slots[] = {
        &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
        &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
    };

    for (int t = 0; t < 7; t++) {
        memcpy(staging + slots[t]->gpu_offset,
               slots[t]->cpu_ptr,
               slots[t]->nbytes);
    }
}

// ============================================================
// Queue background CPU memcpy (non-blocking)
// ============================================================
void LayerStreamer::prefetch_staging(int layer_idx, int slot) {
    if (tiered_mode_ && layer_tier_[layer_idx] != LayerTier::NVME) {
        // VRAM and RAM layers don't need staging
        return;
    }

    if (mmap_pinned_) return;  // No staging needed — direct DMA path

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        const auto& nlay = nvme_layers_[layer_idx];
        gpunvme_err_t err = gpunvme_load_layer(
            &nvme_loader_, nlay.start_lba, nlay.total_bytes, staging_buf_[slot]);

        if (err == GPUNVME_OK) {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
            staging_ready_cv_.notify_all();
            return;
        }
        fprintf(stderr, "LayerStreamer: NVMe read failed for layer %d, fallback to memcpy\n",
                layer_idx);
    }
#endif

    // Fallback: CPU worker thread memcpy
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        staging_ready_[slot] = false;
        worker_request_ = {layer_idx, slot};
        worker_has_work_ = true;
    }
    worker_cv_.notify_one();
}

// ============================================================
// Wait for staging[slot] ready, then issue async H2D
// ============================================================
void LayerStreamer::begin_h2d(int layer_idx, int slot) {
    NT_CHECK(layer_idx >= 0 && layer_idx < (int)layers_.size(), "Layer index out of range");
    NT_CHECK(slot == 0 || slot == 1, "Slot must be 0 or 1");

    // Tier A: VRAM resident — no transfer needed, just record event
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::VRAM) {
        current_layer_[slot] = layer_idx;
        auto& dev = CUDADevice::instance();
        StreamType xfer = (slot == 0) ? STREAM_TRANSFER0 : STREAM_TRANSFER1;
        dev.record_event(transfer_done_[slot], xfer);
        return;
    }

    current_layer_[slot] = layer_idx;

    auto& dev = CUDADevice::instance();
    StreamType xfer = (slot == 0) ? STREAM_TRANSFER0 : STREAM_TRANSFER1;

    // Wait until compute on this slot is done (safe to overwrite GPU buffer)
    dev.wait_event(xfer, compute_done_[slot]);

    uint8_t* gpu_base = static_cast<uint8_t*>(gpu_buf_[slot]);

    // Tier B: async H2D from pinned RAM cache
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::RAM) {
        int ram_idx = layer_idx - tier_config_.n_vram;
        size_t total = layer_transfer_size(layer_idx);
        dev.memcpy_h2d_async(gpu_base, ram_cache_[ram_idx], total, xfer);
        dev.record_event(transfer_done_[slot], xfer);
        return;
    }

    // Tier C / legacy: from mmap or staging buffer
    if (mmap_pinned_) {
        // Direct async copy from pinned mmap to GPU
        const LayerLayout& lay = layers_[layer_idx];
        const TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };
        for (int t = 0; t < 7; t++) {
            dev.memcpy_h2d_async(
                gpu_base + slots[t]->gpu_offset,
                slots[t]->cpu_ptr,
                slots[t]->nbytes,
                xfer
            );
        }
    } else {
        // Wait for worker thread to finish filling staging[slot]
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            staging_ready_cv_.wait(lock, [&] { return staging_ready_[slot]; });
        }

        // Single large async H2D from pinned staging to GPU
        size_t total = layer_transfer_size(layer_idx);
        dev.memcpy_h2d_async(gpu_base, staging_buf_[slot], total, xfer);
    }

    dev.record_event(transfer_done_[slot], xfer);
}

} // namespace nt
