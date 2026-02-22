#include "streamer.h"
#include "../core/device.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace nt {

// ============================================================
// CPU-side FP16 helpers (no CUDA needed)
// ============================================================
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF;
               f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
    else f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}
static uint16_t f32_to_fp16(float val) {
    uint32_t f; memcpy(&f, &val, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;  // flush to zero
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);  // infinity
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// ============================================================
// Q6_K → Q4_K_M in-place requantization
// ============================================================
size_t LayerStreamer::requantize_q6k_to_q4km(void* data, size_t nbytes_q6k) {
    size_t n_blocks = nbytes_q6k / sizeof(BlockQ6_K);
    if (n_blocks == 0 || nbytes_q6k % sizeof(BlockQ6_K) != 0) return nbytes_q6k;

    const uint8_t* src = static_cast<const uint8_t*>(data);
    uint8_t* dst = static_cast<uint8_t*>(data);  // in-place safe: Q4_K (144) < Q6_K (210)

    for (size_t b = 0; b < n_blocks; b++) {
        const BlockQ6_K* q6 = reinterpret_cast<const BlockQ6_K*>(src + b * sizeof(BlockQ6_K));
        BlockQ4_K* q4 = reinterpret_cast<BlockQ4_K*>(dst + b * sizeof(BlockQ4_K));

        // Step 1: Dequantize Q6_K block → 256 floats
        float vals[256];
        float d = fp16_to_f32(q6->d);
        const uint8_t* ql = q6->ql;
        const uint8_t* qh = q6->qh;
        const int8_t*  sc = q6->scales;
        float* y = vals;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l]     & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l+32]  & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l]      >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4i = (int)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l]      = d * (float)sc[is + 0] * q1;
                y[l + 32] = d * (float)sc[is + 2] * q2;
                y[l + 64] = d * (float)sc[is + 4] * q3;
                y[l + 96] = d * (float)sc[is + 6] * q4i;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }

        // Step 2: Quantize 256 floats → Q4_K_M block
        // 8 sub-blocks of 32 values each
        float sub_scales[8], sub_mins[8];
        for (int sb = 0; sb < 8; sb++) {
            float mn = vals[sb * 32], mx = vals[sb * 32];
            for (int j = 1; j < 32; j++) {
                float v = vals[sb * 32 + j];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            sub_scales[sb] = (mx - mn) / 15.0f;
            sub_mins[sb] = -mn;
            if (sub_scales[sb] < 1e-10f) sub_scales[sb] = 1e-10f;
        }

        // Find super-block scales
        float max_scale = 0.0f, max_min = 0.0f;
        for (int sb = 0; sb < 8; sb++) {
            if (sub_scales[sb] > max_scale) max_scale = sub_scales[sb];
            if (sub_mins[sb] > max_min) max_min = sub_mins[sb];
        }

        q4->d    = f32_to_fp16(max_scale > 0 ? max_scale / 63.0f : 0.0f);
        q4->dmin = f32_to_fp16(max_min > 0 ? max_min / 63.0f : 0.0f);

        float inv_d    = max_scale > 0 ? 63.0f / max_scale : 0.0f;
        float inv_dmin = max_min > 0 ? 63.0f / max_min : 0.0f;

        // Compute and pack 6-bit sub-block scales and mins
        uint8_t sc6[8], m6[8];
        for (int sb = 0; sb < 8; sb++) {
            int s = (int)(sub_scales[sb] * inv_d + 0.5f);
            int m = (int)(sub_mins[sb] * inv_dmin + 0.5f);
            sc6[sb] = (uint8_t)(s < 0 ? 0 : s > 63 ? 63 : s);
            m6[sb]  = (uint8_t)(m < 0 ? 0 : m > 63 ? 63 : m);
        }

        // Pack scales[12] (Q4_K_M format)
        for (int sb = 0; sb < 4; sb++) {
            q4->scales[sb]     = sc6[sb] | ((sc6[sb + 4] & 0x30) << 2);
            q4->scales[sb + 4] = m6[sb]  | ((m6[sb + 4] & 0x30) << 2);
        }
        q4->scales[8]  = (sc6[4] & 0x0F) | ((m6[4] & 0x0F) << 4);
        q4->scales[9]  = (sc6[5] & 0x0F) | ((m6[5] & 0x0F) << 4);
        q4->scales[10] = (sc6[6] & 0x0F) | ((m6[6] & 0x0F) << 4);
        q4->scales[11] = (sc6[7] & 0x0F) | ((m6[7] & 0x0F) << 4);

        // Reconstruct actual scales for quantization
        float actual_d = fp16_to_f32(q4->d);
        float actual_dmin = fp16_to_f32(q4->dmin);

        // Quantize each sub-block into 4-bit nibbles
        memset(q4->qs, 0, 128);
        for (int sb = 0; sb < 8; sb++) {
            float sd = actual_d * sc6[sb];
            float sm = actual_dmin * m6[sb];
            float inv_sd = sd > 1e-10f ? 1.0f / sd : 0.0f;

            for (int j = 0; j < 16; j++) {
                // Each byte stores two nibbles: weights j and j+16 of the sub-block
                float v_lo = vals[sb * 32 + j];
                float v_hi = vals[sb * 32 + j + 16];

                int q_lo = (int)((v_lo + sm) * inv_sd + 0.5f);
                int q_hi = (int)((v_hi + sm) * inv_sd + 0.5f);
                q_lo = q_lo < 0 ? 0 : q_lo > 15 ? 15 : q_lo;
                q_hi = q_hi < 0 ? 0 : q_hi > 15 ? 15 : q_hi;

                q4->qs[sb * 16 + j] = (uint8_t)(q_lo | (q_hi << 4));
            }
        }
    }

    return n_blocks * sizeof(BlockQ4_K);
}

// ============================================================
// PCIe bandwidth detection via sysfs
//
// Prefers max_link_speed / max_link_width over current_link_* because
// PCIe power management (ASPM) downgrades the link to Gen1/2 at idle,
// causing wildly incorrect bandwidth estimates. The max values reflect
// what the slot actually negotiated at boot time and stays stable.
//
// Falls back to current_link_* if max_* is unavailable (older kernels).
//
// Returns effective one-directional bandwidth in GB/s,
// or 0.0 if detection fails (caller will use safe default).
// ============================================================
static float detect_pcie_bandwidth_gbps() {
    // Retrieve CUDA device PCI bus ID (e.g. "0000:05:00.0")
    char pci_id[32] = {};
    if (cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), 0) != cudaSuccess)
        return 0.0f;

    // sysfs paths use lowercase hex
    for (int i = 0; pci_id[i]; i++)
        pci_id[i] = (char)tolower((unsigned char)pci_id[i]);

    char path[256];
    float speed_gts = 0.0f;
    int width = 0;

    // Helper: try max_link_* first (immune to ASPM idle downclocking),
    // fall back to current_link_* for older kernels that lack max_*.
    auto read_speed = [&](const char* attr_max, const char* attr_cur) -> float {
        float val = 0.0f;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", pci_id, attr_max);
        FILE* f = fopen(path, "r");
        if (f) { fscanf(f, "%f", &val); fclose(f); }
        if (val > 0.0f) return val;
        // fallback
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", pci_id, attr_cur);
        f = fopen(path, "r");
        if (f) { fscanf(f, "%f", &val); fclose(f); }
        return val;
    };
    auto read_width = [&](const char* attr_max, const char* attr_cur) -> int {
        int val = 0;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", pci_id, attr_max);
        FILE* f = fopen(path, "r");
        if (f) { fscanf(f, "%d", &val); fclose(f); }
        if (val > 0) return val;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/%s", pci_id, attr_cur);
        f = fopen(path, "r");
        if (f) { fscanf(f, "%d", &val); fclose(f); }
        return val;
    };

    speed_gts = read_speed("max_link_speed", "current_link_speed");
    width     = read_width("max_link_width", "current_link_width");

    if (speed_gts <= 0.0f || width <= 0) return 0.0f;

    // Convert GT/s × width to effective GB/s:
    //   Gen1/2: 8b/10b encoding → 80% efficiency
    //   Gen3+:  128b/130b encoding → ~98.5% efficiency
    //   Additional 0.985 factor for protocol overhead.
    float enc = (speed_gts <= 5.0f) ? 0.8f : 0.985f;
    return speed_gts / 8.0f * enc * (float)width * 0.985f;
}

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

    // Query RAM: both total (for adaptive reserve) and available
    size_t ram_total = 0;
    size_t ram_free  = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            size_t val;
            if (sscanf(line, "MemTotal: %zu kB", &val) == 1)
                ram_total = val * 1024;
            else if (sscanf(line, "MemAvailable: %zu kB", &val) == 1)
                ram_free = val * 1024;
        }
        fclose(f);
    }
    if (ram_free == 0) {
        // Fallback to sysinfo if /proc/meminfo unavailable
        struct sysinfo si;
        sysinfo(&si);
        ram_free  = (size_t)si.freeram  * si.mem_unit
                  + (size_t)si.bufferram * si.mem_unit;
        ram_total = (size_t)si.totalram * si.mem_unit;
    }

    // Adaptive RAM reserve: max(4 GB, 15% of total RAM).
    // This ensures the OS always retains enough memory on any machine size,
    // rather than using a fixed 6 GB that wastes memory on small machines
    // or under-reserves on large ones.
    if (ram_reserve == 0) {
        const size_t min_reserve  = 4ULL << 30;            // 4 GB floor
        const size_t pct_reserve  = ram_total * 15 / 100;  // 15% of total
        ram_reserve = (pct_reserve > min_reserve) ? pct_reserve : min_reserve;
    }

    // Detect PCIe bandwidth and log the result
    float pcie_bw = detect_pcie_bandwidth_gbps();
    if (pcie_bw > 0.0f) {
        // Determine generation label for the log message
        const char* gen_label = "Gen3";
        if      (pcie_bw >= 120.0f) gen_label = "Gen6";
        else if (pcie_bw >=  60.0f) gen_label = "Gen5";
        else if (pcie_bw >=  30.0f) gen_label = "Gen4";
        else if (pcie_bw >=  15.0f) gen_label = "Gen3";
        else                         gen_label = "Gen1/2";
        fprintf(stderr, "TierConfig: PCIe %s x%d = %.1f GB/s (detected)\n",
                gen_label,
                (int)roundf(pcie_bw / (pcie_bw >= 60.0f ? 7.876f :
                             pcie_bw >= 30.0f ? 3.938f :
                             pcie_bw >= 15.0f ? 1.970f : 0.985f)),
                pcie_bw);
    } else {
        fprintf(stderr, "TierConfig: PCIe detection failed — defaulting to 16.0 GB/s\n");
        pcie_bw = 16.0f;
    }
    tc.pcie_bandwidth_gbps = pcie_bw;

    fprintf(stderr, "TierConfig: VRAM free=%.1f GB, RAM available=%.1f GB\n",
            vram_free / (1024.0 * 1024 * 1024), ram_free / (1024.0 * 1024 * 1024));
    fprintf(stderr, "TierConfig: VRAM reserve=%.1f GB, RAM reserve=%.1f GB (adaptive)\n",
            vram_reserve / (1024.0 * 1024 * 1024), ram_reserve / (1024.0 * 1024 * 1024));

    size_t vram_avail = (vram_free > vram_reserve) ? (vram_free - vram_reserve) : 0;
    size_t ram_avail  = (ram_free > ram_reserve) ? (ram_free - ram_reserve) : 0;

    tc.n_vram = std::min(n_layers, (int)(vram_avail / layer_bytes));
    int remaining = n_layers - tc.n_vram;
    tc.n_ram  = std::min(remaining, (int)(ram_avail / layer_bytes));

    // Allow env var overrides to cap tier A/B (forces layers to tier C for testing)
    const char* max_vram_str = getenv("GPUNVME_MAX_VRAM_LAYERS");
    const char* max_ram_str  = getenv("GPUNVME_MAX_RAM_LAYERS");
    if (max_vram_str) {
        int cap = atoi(max_vram_str);
        if (cap >= 0 && cap < tc.n_vram) {
            fprintf(stderr, "TierConfig: GPUNVME_MAX_VRAM_LAYERS=%d (was %d)\n", cap, tc.n_vram);
            tc.n_vram = cap;
        }
    }
    if (max_ram_str) {
        int cap = atoi(max_ram_str);
        remaining = n_layers - tc.n_vram;
        if (cap >= 0 && cap < tc.n_ram) {
            fprintf(stderr, "TierConfig: GPUNVME_MAX_RAM_LAYERS=%d (was %d)\n", cap, tc.n_ram);
            tc.n_ram = std::min(remaining, cap);
        }
    }

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
// TierConfig::optimal_pipeline_depth
//
// Returns the recommended number of streaming buffers based on
// measured PCIe bandwidth:
//   ≥63 GB/s  (Gen5 x16 or Gen4 x32) → 3 slots: compute can
//              overlap two full H2D transfers simultaneously
//   <63 GB/s  (Gen4 x16 and below)   → 2 slots: classic
//              double-buffer is sufficient
// ============================================================
int TierConfig::optimal_pipeline_depth() const {
    // 63 GB/s threshold cleanly separates Gen5 x16 (≈63.0) from
    // Gen4 x16 (≈31.5). Gen4 x32 (≈63.0) also benefits from 3 slots.
    return (pcie_bandwidth_gbps >= 63.0f) ? 3 : 2;
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

            slots[t]->gpu_offset    = offset;
            slots[t]->cpu_ptr       = loader.tensor_data(name);
            slots[t]->nbytes        = info->nbytes;
            slots[t]->xfer_nbytes   = info->nbytes;  // may be reduced by requantization
            slots[t]->dtype         = ggml_to_dtype(info->ggml_type);

            NT_CHECK(slots[t]->cpu_ptr != nullptr, ("Null data for tensor: " + name).c_str());

            offset += info->nbytes;
        }

        size_t layer_bytes = (offset + 255) & ~(size_t)255;
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }

    buf_size_ = max_layer_bytes;

    // --------------------------------------------------------
    // Auto-select pipeline depth if not manually overridden.
    // Detect PCIe bandwidth and pick 3 slots for Gen5, 2 for
    // Gen4 and below. NT_PIPELINE_DEPTH env var overrides both.
    // --------------------------------------------------------
    const char* env_depth = getenv("NT_PIPELINE_DEPTH");
    if (env_depth) {
        int n = atoi(env_depth);
        if (n >= 2 && n <= 8 && n != n_slots_) {
            fprintf(stderr, "LayerStreamer: NT_PIPELINE_DEPTH=%d override\n", n);
            n_slots_ = n;
        }
    }
    if (n_slots_ == 0) {
        float pcie_bw = detect_pcie_bandwidth_gbps();
        if (pcie_bw > 0.0f) {
            n_slots_ = (pcie_bw >= 63.0f) ? 3 : 2;
            fprintf(stderr, "Pipeline depth: %d (PCIe %.1f GB/s autodetect)\n",
                    n_slots_, pcie_bw);
        } else {
            n_slots_ = 2;
            fprintf(stderr, "Pipeline depth: %d (PCIe detection failed, default)\n", n_slots_);
        }
    } else {
        fprintf(stderr, "Pipeline depth: %d (manual)\n", n_slots_);
    }

    fprintf(stderr, "LayerStreamer: %d layers, buffer size: %.1f MB each (%.1f MB total for %d buffers)\n",
        n_layers, buf_size_ / (1024.0 * 1024.0),
        n_slots_ * buf_size_ / (1024.0 * 1024.0), n_slots_);

    // Allocate n_slots_ GPU buffers and CUDA events
    gpu_buf_.resize(n_slots_, nullptr);
    current_layer_.resize(n_slots_, -1);
    transfer_done_.resize(n_slots_, nullptr);
    compute_done_.resize(n_slots_, nullptr);

    for (int s = 0; s < n_slots_; s++) {
        cudaError_t err = cudaMalloc(&gpu_buf_[s], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate GPU layer buffer");
        current_layer_[s] = -1;
    }

    // Create CUDA events (disable timing for lower overhead)
    for (int s = 0; s < n_slots_; s++) {
        cudaEvent_t ev;
        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        transfer_done_[s] = static_cast<void*>(ev);

        NT_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        compute_done_[s] = static_cast<void*>(ev);
    }

    // Try to pin the mmap'd region for true async DMA
    // Skip pinning for large models (>50% of RAM) — pinning consumes physical RAM
    // and leaves nothing for tier B layer caches. Staging buffers only use ~1GB.
    const void* data_ptr = loader.mmap_data_ptr();
    size_t data_size = loader.tensor_data_size();

    size_t total_ram = 0;
    FILE* mi = fopen("/proc/meminfo", "r");
    if (mi) {
        char line[256];
        while (fgets(line, sizeof(line), mi)) {
            size_t val;
            if (sscanf(line, "MemTotal: %zu kB", &val) == 1) {
                total_ram = val * 1024;
                break;
            }
        }
        fclose(mi);
    }

    bool skip_pin = (total_ram > 0 && data_size > total_ram / 2);
    cudaError_t pin_err = cudaErrorMemoryAllocation;  // default to "failed"

    if (!skip_pin) {
        pin_err = cudaHostRegister(
            const_cast<void*>(data_ptr), data_size, cudaHostRegisterReadOnly);
    } else {
        fprintf(stderr, "LayerStreamer: skipping mmap pin (%.1f GB > 50%% of %.1f GB RAM) — preserving RAM for tier B\n",
            data_size / (1024.0 * 1024.0 * 1024.0), total_ram / (1024.0 * 1024.0 * 1024.0));
    }

    if (pin_err == cudaSuccess) {
        mmap_pinned_ = true;
        fprintf(stderr, "LayerStreamer: mmap region pinned (%.1f GB) — true async DMA enabled\n",
            data_size / (1024.0 * 1024.0 * 1024.0));
    } else {
        mmap_pinned_ = false;
        fprintf(stderr, "LayerStreamer: cudaHostRegister failed (%s), using double pinned staging buffers\n",
            cudaGetErrorString(pin_err));

        // Allocate n_slots_ pinned staging buffers for pipelined overlap
        staging_size_ = buf_size_;
        staging_buf_.resize(n_slots_, nullptr);
        staging_ready_.resize(n_slots_, 0);
        for (int s = 0; s < n_slots_; s++) {
            cudaError_t err = cudaMallocHost(&staging_buf_[s], staging_size_);
            NT_CHECK(err == cudaSuccess, "Failed to allocate pinned staging buffer");
            staging_ready_[s] = 0;
        }
        fprintf(stderr, "LayerStreamer: pinned staging: %.1f MB x %d = %.1f MB\n",
            staging_size_ / (1024.0 * 1024.0), n_slots_,
            n_slots_ * staging_size_ / (1024.0 * 1024.0));

        // Start worker thread for background CPU memcpy
        worker_shutdown_ = false;
        worker_has_work_ = false;
        worker_thread_ = std::thread(&LayerStreamer::worker_loop, this);
        fprintf(stderr, "LayerStreamer: worker thread started\n");
    }

    // Record initial compute_done events so the first begin_transfer doesn't deadlock
    auto& dev = CUDADevice::instance();
    for (int s = 0; s < n_slots_; s++) {
        dev.record_event(compute_done_[s], STREAM_COMPUTE);
    }

#ifdef USE_GPUNVME
    // Try to initialize gpu-nvme-direct Layer Loader
    const char* nvme_bdf = getenv("GPUNVME_PCI_BDF");
    const char* nvme_lba_str = getenv("GPUNVME_GGUF_LBA");

    if (nvme_bdf && nvme_lba_str) {
        uint64_t gguf_start_lba = strtoull(nvme_lba_str, NULL, 0);

        // Find the maximum file span across all layers for buffer sizing
        size_t max_file_span = 0;
        for (int i = 0; i < n_layers; i++) {
            uint64_t min_off = UINT64_MAX, max_end = 0;
            for (int t = 0; t < 7; t++) {
                std::string name = "blk." + std::to_string(i) + "." + tensor_suffixes[t];
                uint64_t off = loader.tensor_file_offset(name);
                const GGUFTensorInfo* info = loader.tensor_info(name);
                if (off < min_off) min_off = off;
                if (off + info->nbytes > max_end) max_end = off + info->nbytes;
            }
            size_t span = max_end - (min_off & ~(size_t)511);  // LBA-aligned start
            if (span > max_file_span) max_file_span = span;
        }
        // Round up to block size
        max_file_span = (max_file_span + 511) & ~(size_t)511;

        gpunvme_err_t err = gpunvme_layer_loader_init(
            &nvme_loader_, nvme_bdf, max_file_span, /*pipeline_depth=*/32);

        if (err == GPUNVME_OK) {
            nvme_initialized_ = true;
            gguf_start_lba_ = gguf_start_lba;
            nvme_block_size_ = gpunvme_layer_loader_block_size(&nvme_loader_);

            // Allocate NVMe read buffer (pinned for potential reuse)
            nvme_read_buf_size_ = max_file_span;
            cudaError_t cerr = cudaMallocHost(&nvme_read_buf_, nvme_read_buf_size_);
            if (cerr != cudaSuccess) {
                fprintf(stderr, "LayerStreamer: NVMe read buffer alloc failed, disabling NVMe\n");
                gpunvme_layer_loader_destroy(&nvme_loader_);
                nvme_initialized_ = false;
            }
        }

        if (nvme_initialized_) {
            // Pre-compute per-layer NVMe read info with scatter-copy mapping
            nvme_layers_.resize(n_layers);
            for (int i = 0; i < n_layers; i++) {
                std::string pfx = "blk." + std::to_string(i) + ".";
                const LayerLayout& lay = layers_[i];
                const TensorSlot* slots[] = {
                    &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
                    &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
                };

                uint64_t min_off = UINT64_MAX, max_end = 0;
                uint64_t tensor_file_offs[7];
                for (int t = 0; t < 7; t++) {
                    std::string name = pfx + tensor_suffixes[t];
                    tensor_file_offs[t] = loader.tensor_file_offset(name);
                    size_t sz = slots[t]->nbytes;
                    if (tensor_file_offs[t] < min_off) min_off = tensor_file_offs[t];
                    if (tensor_file_offs[t] + sz > max_end) max_end = tensor_file_offs[t] + sz;
                }

                // LBA-align the span
                uint64_t span_start = (min_off / nvme_block_size_) * nvme_block_size_;
                uint64_t span_end = ((max_end + nvme_block_size_ - 1) / nvme_block_size_) * nvme_block_size_;

                nvme_layers_[i].start_lba = gguf_start_lba_ + span_start / nvme_block_size_;
                nvme_layers_[i].read_bytes = span_end - span_start;

                for (int t = 0; t < 7; t++) {
                    nvme_layers_[i].tensors[t].read_offset = tensor_file_offs[t] - span_start;
                    nvme_layers_[i].tensors[t].gpu_offset = slots[t]->gpu_offset;
                    nvme_layers_[i].tensors[t].nbytes = slots[t]->nbytes;

                    // Per-tensor LBA info (for BAR1 scatter reads)
                    uint64_t t_file_start = (tensor_file_offs[t] / nvme_block_size_) * nvme_block_size_;
                    uint64_t t_file_end = ((tensor_file_offs[t] + slots[t]->nbytes + nvme_block_size_ - 1)
                                           / nvme_block_size_) * nvme_block_size_;
                    nvme_layers_[i].tensors[t].start_lba = gguf_start_lba_ + t_file_start / nvme_block_size_;
                    nvme_layers_[i].tensors[t].lba_aligned_bytes = t_file_end - t_file_start;
                    nvme_layers_[i].tensors[t].lba_sub_offset = tensor_file_offs[t] - t_file_start;
                }
            }

            fprintf(stderr, "LayerStreamer: NVMe backend OK (MDTS=%uK, block=%u, read_buf=%.1f MB)\n",
                    gpunvme_layer_loader_max_transfer(&nvme_loader_) / 1024,
                    nvme_block_size_,
                    nvme_read_buf_size_ / (1024.0 * 1024));

            // Try BAR1 direct VRAM mode
            const char* gpu_bdf = getenv("GPUNVME_GPU_BDF");
            if (!gpu_bdf) gpu_bdf = "0000:0a:00.0";  // default for our RTX 3090

            gpunvme_err_t bar1_err = gpunvme_bar1_init(&nvme_loader_, gpu_bdf, 0x20000000ULL);
            if (bar1_err == GPUNVME_OK) {
                fprintf(stderr, "LayerStreamer: BAR1 init OK — Tier 2 NVMe→VRAM available\n");
            } else {
                fprintf(stderr, "LayerStreamer: BAR1 init failed (err=%d), using Tier 1 (NVMe→host)\n",
                        bar1_err);
            }
        }

        if (!nvme_initialized_ && err != GPUNVME_OK) {
            fprintf(stderr, "LayerStreamer: NVMe init failed (err=%d), fallback to mmap\n", err);
        }
    }

    // If BAR1 available, resolve GPU buffer physical addresses
    if (nvme_initialized_ && nvme_loader_.bar1_enabled) {
        bool both_ok = true;
        for (int s = 0; s < 2; s++) {
            gpunvme_err_t r = gpunvme_bar1_resolve(&nvme_loader_, gpu_buf_[s], buf_size_, &bar1_phys_[s]);
            if (r != GPUNVME_OK) {
                fprintf(stderr, "LayerStreamer: BAR1 resolve failed for slot %d\n", s);
                both_ok = false;
                break;
            }
        }
        if (both_ok) {
            // Allocate VRAM temp buffer for BAR1 bulk reads
            nvme_vram_temp_size_ = nvme_read_buf_size_;  // = max_file_span
            cudaError_t terr = cudaMalloc(&nvme_vram_temp_, nvme_vram_temp_size_);
            if (terr == cudaSuccess) {
                gpunvme_err_t r = gpunvme_bar1_resolve(
                    &nvme_loader_, nvme_vram_temp_, nvme_vram_temp_size_,
                    &nvme_vram_temp_bar1_phys_);
                if (r == GPUNVME_OK) {
                    bar1_enabled_ = true;
                    fprintf(stderr, "LayerStreamer: BAR1 Tier 2 + bulk temp (%.1f MB VRAM)\n",
                            nvme_vram_temp_size_ / (1024.0 * 1024));
                } else {
                    cudaFree(nvme_vram_temp_);
                    nvme_vram_temp_ = nullptr;
                }
            }
            // If temp alloc/resolve failed, BAR1 stays disabled
            if (!nvme_vram_temp_) bar1_enabled_ = false;
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
    // KV cache: 2 (K+V) × n_layers × max_seq × n_kv_heads × head_dim × sizeof(half)
    size_t kv_bytes = 2ULL * n_layers * config.max_seq_len * config.n_kv_heads
                      * config.head_dim * sizeof(float16_t);
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
    size_t requant_saved = 0;
    for (int i = 0; i < tier_config_.n_ram; i++) {
        int layer_idx = tier_config_.n_vram + i;
        cudaError_t err = cudaMallocHost(&ram_cache_[i], buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate pinned RAM cache buffer");

        // Copy all 7 tensors from mmap to pinned RAM
        LayerLayout& lay = layers_[layer_idx];
        TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };
        uint8_t* ram_base = static_cast<uint8_t*>(ram_cache_[i]);
        for (int t = 0; t < 7; t++) {
            memcpy(ram_base + slots[t]->gpu_offset,
                   slots[t]->cpu_ptr, slots[t]->nbytes);

            // Requantize Q6_K → Q4_K_M in-place if enabled
            if (requant_tier_b_ && slots[t]->dtype == DType::Q6_K) {
                size_t new_bytes = requantize_q6k_to_q4km(
                    ram_base + slots[t]->gpu_offset, slots[t]->nbytes);
                requant_saved += slots[t]->nbytes - new_bytes;
                slots[t]->xfer_nbytes = new_bytes;
                slots[t]->dtype = DType::Q4_K_M;
            }
        }

        if ((i + 1) % 10 == 0 || i == tier_config_.n_ram - 1) {
            fprintf(stderr, "  Loaded tier B layer %d/%d to pinned RAM%s\n",
                    i + 1, tier_config_.n_ram,
                    requant_tier_b_ ? " (Q4_K_M)" : "");
        }
    }

    if (requant_tier_b_ && requant_saved > 0) {
        fprintf(stderr, "LayerStreamer: requantized Q6_K→Q4_K_M: saved %.1f MB H2D/layer\n",
                (float)requant_saved / tier_config_.n_ram / (1024.0 * 1024));
    }

    tiered_mode_ = true;

    fprintf(stderr, "LayerStreamer: tiered init complete\n");
    fprintf(stderr, "  Free VRAM: %.1f GB\n",
            CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));
}

// ============================================================
// Delta encoding: init from .ntd file
// ============================================================
bool LayerStreamer::init_delta(const std::string& ntd_path, const ModelConfig& config) {
    // Open and mmap the .ntd file
    delta_mmap_fd_ = open(ntd_path.c_str(), O_RDONLY);
    if (delta_mmap_fd_ < 0) {
        fprintf(stderr, "init_delta: cannot open %s\n", ntd_path.c_str());
        return false;
    }

    struct stat st;
    fstat(delta_mmap_fd_, &st);
    delta_mmap_size_ = st.st_size;

    delta_mmap_ = mmap(nullptr, delta_mmap_size_, PROT_READ, MAP_PRIVATE, delta_mmap_fd_, 0);
    if (delta_mmap_ == MAP_FAILED) {
        fprintf(stderr, "init_delta: mmap failed\n");
        close(delta_mmap_fd_);
        delta_mmap_fd_ = -1;
        return false;
    }

    // Parse header (64 bytes)
    const uint8_t* hdr = static_cast<const uint8_t*>(delta_mmap_);
    if (memcmp(hdr, "NTD1", 4) != 0) {
        fprintf(stderr, "init_delta: bad magic\n");
        munmap(delta_mmap_, delta_mmap_size_);
        close(delta_mmap_fd_);
        return false;
    }

    uint32_t rank, n_layers, hidden_size, intermediate_size, n_heads, n_kv_heads, head_dim;
    uint32_t base_dtype_id, delta_dtype_id;
    uint64_t base_offset, delta_offset;

    memcpy(&rank, hdr + 4, 4);
    memcpy(&n_layers, hdr + 8, 4);
    memcpy(&hidden_size, hdr + 12, 4);
    memcpy(&intermediate_size, hdr + 16, 4);
    memcpy(&n_heads, hdr + 20, 4);
    memcpy(&n_kv_heads, hdr + 24, 4);
    memcpy(&head_dim, hdr + 28, 4);
    memcpy(&base_dtype_id, hdr + 32, 4);
    memcpy(&delta_dtype_id, hdr + 36, 4);
    memcpy(&base_offset, hdr + 40, 8);
    memcpy(&delta_offset, hdr + 48, 8);

    delta_rank_ = rank;

    // Verify config matches
    if ((int)n_layers != config.n_layers || (int)hidden_size != config.hidden_size) {
        fprintf(stderr, "init_delta: config mismatch (layers %u vs %d, hidden %u vs %d)\n",
                n_layers, config.n_layers, hidden_size, config.hidden_size);
        munmap(delta_mmap_, delta_mmap_size_);
        close(delta_mmap_fd_);
        return false;
    }

    fprintf(stderr, "init_delta: rank=%d, n_layers=%d, hidden=%d, intermediate=%d\n",
            rank, n_layers, hidden_size, intermediate_size);

    // Compute base weight sizes (7 Q6_K matrices)
    // Weight shapes: [out, in] — same as compute_weight_shapes in decompose tool
    int h = hidden_size;
    int inter = intermediate_size;
    int nh = n_heads;
    int nkv = n_kv_heads;
    int hd_val = head_dim;

    struct { int out, in; } base_shapes[7] = {
        {(int)(nh * hd_val), h},     // attn_q
        {(int)(nkv * hd_val), h},    // attn_k
        {(int)(nkv * hd_val), h},    // attn_v
        {h, (int)(nh * hd_val)},     // attn_o
        {inter, h},                  // ffn_gate
        {inter, h},                  // ffn_up
        {h, inter},                  // ffn_down
    };

    // Compute base section layout
    size_t base_cursor = 0;
    for (int w = 0; w < 7; w++) {
        delta_base_offsets_[w] = base_cursor;
        delta_base_dtypes_[w] = DType::Q6_K;  // base is always Q6_K
        size_t n_elements = (size_t)base_shapes[w].out * base_shapes[w].in;
        size_t bytes = (n_elements / 256) * 210;  // Q6_K: 256 weights per 210-byte block
        base_cursor += bytes;
        // Align to 256 bytes
        base_cursor = (base_cursor + 255) & ~(size_t)255;
    }
    delta_base_size_ = base_cursor;

    fprintf(stderr, "init_delta: base size = %.1f MB\n",
            delta_base_size_ / (1024.0 * 1024.0));

    // Allocate VRAM for base weights and copy from mmap
    cudaError_t err = cudaMalloc(&delta_base_buf_, delta_base_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "init_delta: failed to allocate base VRAM (%.1f MB)\n",
                delta_base_size_ / (1024.0 * 1024.0));
        munmap(delta_mmap_, delta_mmap_size_);
        close(delta_mmap_fd_);
        return false;
    }

    // Copy base weights: file offset = base_offset, sequential Q6_K data
    {
        const uint8_t* base_src = static_cast<const uint8_t*>(delta_mmap_) + base_offset;
        size_t file_cursor = 0;
        for (int w = 0; w < 7; w++) {
            size_t n_elements = (size_t)base_shapes[w].out * base_shapes[w].in;
            size_t bytes = (n_elements / 256) * 210;
            uint8_t* dst = static_cast<uint8_t*>(delta_base_buf_) + delta_base_offsets_[w];
            nt_cuda_memcpy_h2d(dst, base_src + file_cursor, bytes);
            file_cursor += bytes;
        }
        fprintf(stderr, "init_delta: base weights loaded to VRAM\n");
    }

    // Compute per-layer delta layout in the .ntd file
    delta_layers_.resize(n_layers);
    size_t file_cursor = delta_offset;
    for (int layer = 0; layer < (int)n_layers; layer++) {
        for (int w = 0; w < 7; w++) {
            // U: [out_features, rank] F16
            size_t u_bytes = (size_t)base_shapes[w].out * rank * 2;
            delta_layers_[layer].U[w] = {file_cursor, u_bytes};
            file_cursor += u_bytes;

            // V: [rank, in_features] F16
            size_t v_bytes = (size_t)rank * base_shapes[w].in * 2;
            delta_layers_[layer].V[w] = {file_cursor, v_bytes};
            file_cursor += v_bytes;
        }
    }

    // Compute per-slot GPU buffer layout for delta tensors
    // All 14 tensors (7 U + 7 V) packed sequentially with 256-byte alignment
    size_t slot_cursor = 0;
    for (int w = 0; w < 7; w++) {
        delta_slot_layout_.U_offset[w] = slot_cursor;
        size_t u_bytes = (size_t)base_shapes[w].out * rank * 2;
        slot_cursor += u_bytes;
        slot_cursor = (slot_cursor + 255) & ~(size_t)255;

        delta_slot_layout_.V_offset[w] = slot_cursor;
        size_t v_bytes = (size_t)rank * base_shapes[w].in * 2;
        slot_cursor += v_bytes;
        slot_cursor = (slot_cursor + 255) & ~(size_t)255;
    }
    delta_buf_size_ = slot_cursor;

    fprintf(stderr, "init_delta: delta per-layer = %.1f MB (vs %.1f MB full layer)\n",
            delta_buf_size_ / (1024.0 * 1024.0), buf_size_ / (1024.0 * 1024.0));

    // Reallocate GPU buffers to delta size (much smaller) for all n_slots_
    for (int s = 0; s < n_slots_; s++) {
        if (gpu_buf_[s]) cudaFree(gpu_buf_[s]);
        err = cudaMalloc(&gpu_buf_[s], delta_buf_size_);
        NT_CHECK(err == cudaSuccess, "Failed to allocate delta GPU buffer");
    }

    // Reallocate staging buffers if needed
    if (!mmap_pinned_ && !staging_buf_.empty() && staging_buf_[0]) {
        for (int s = 0; s < n_slots_; s++) {
            cudaFreeHost(staging_buf_[s]);
            err = cudaMallocHost(&staging_buf_[s], delta_buf_size_);
            NT_CHECK(err == cudaSuccess, "Failed to allocate delta staging buffer");
        }
        staging_size_ = delta_buf_size_;
    }

    // Allocate temp buffer for rank-sized intermediate (rank floats)
    err = cudaMalloc(reinterpret_cast<void**>(&delta_temp_), rank * sizeof(float));
    NT_CHECK(err == cudaSuccess, "Failed to allocate delta temp buffer");

    // Pin the mmap region for direct DMA
    cudaError_t pin_err = cudaHostRegister(
        delta_mmap_, delta_mmap_size_, cudaHostRegisterReadOnly);
    if (pin_err == cudaSuccess) {
        mmap_pinned_ = true;
        fprintf(stderr, "init_delta: .ntd mmap pinned (%.1f MB)\n",
                delta_mmap_size_ / (1024.0 * 1024.0));
    }

    delta_mode_ = true;
    buf_size_ = delta_buf_size_;  // update for buffer_size() queries

    fprintf(stderr, "init_delta: ready (rank=%d, %.1f MB base + %.1f MB/layer deltas)\n",
            rank, delta_base_size_ / (1024.0 * 1024.0), delta_buf_size_ / (1024.0 * 1024.0));
    return true;
}

// ============================================================
// Delta: get base weight pointer by index
// ============================================================
const void* LayerStreamer::base_weight_ptr(int weight_idx) const {
    NT_CHECK(delta_mode_ && weight_idx >= 0 && weight_idx < 7,
             "base_weight_ptr: invalid state or index");
    return static_cast<const uint8_t*>(delta_base_buf_) + delta_base_offsets_[weight_idx];
}

// ============================================================
// Delta: get U/V pointers from GPU buffer slot
// ============================================================
DeltaWeightPtrs LayerStreamer::get_delta_weights(int slot) const {
    NT_CHECK(delta_mode_ && slot >= 0 && slot < n_slots_, "get_delta_weights: bad state/slot");
    const uint8_t* base = static_cast<const uint8_t*>(gpu_buf_[slot]);

    DeltaWeightPtrs dp;
    dp.attn_q_U   = base + delta_slot_layout_.U_offset[0];
    dp.attn_q_V   = base + delta_slot_layout_.V_offset[0];
    dp.attn_k_U   = base + delta_slot_layout_.U_offset[1];
    dp.attn_k_V   = base + delta_slot_layout_.V_offset[1];
    dp.attn_v_U   = base + delta_slot_layout_.U_offset[2];
    dp.attn_v_V   = base + delta_slot_layout_.V_offset[2];
    dp.attn_o_U   = base + delta_slot_layout_.U_offset[3];
    dp.attn_o_V   = base + delta_slot_layout_.V_offset[3];
    dp.ffn_gate_U = base + delta_slot_layout_.U_offset[4];
    dp.ffn_gate_V = base + delta_slot_layout_.V_offset[4];
    dp.ffn_up_U   = base + delta_slot_layout_.U_offset[5];
    dp.ffn_up_V   = base + delta_slot_layout_.V_offset[5];
    dp.ffn_down_U = base + delta_slot_layout_.U_offset[6];
    dp.ffn_down_V = base + delta_slot_layout_.V_offset[6];
    return dp;
}

// ============================================================
// Shutdown: free everything
// ============================================================
void LayerStreamer::shutdown() {
    // Free delta resources
    if (delta_base_buf_) { cudaFree(delta_base_buf_); delta_base_buf_ = nullptr; }
    if (delta_temp_) { cudaFree(delta_temp_); delta_temp_ = nullptr; }
    if (delta_mmap_) {
        munmap(delta_mmap_, delta_mmap_size_);
        delta_mmap_ = nullptr;
    }
    if (delta_mmap_fd_ >= 0) { close(delta_mmap_fd_); delta_mmap_fd_ = -1; }
    delta_layers_.clear();
    delta_mode_ = false;

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
    if (nvme_vram_temp_) {
        cudaFree(nvme_vram_temp_);
        nvme_vram_temp_ = nullptr;
        nvme_vram_temp_size_ = 0;
    }
    if (nvme_read_buf_) {
        cudaFreeHost(nvme_read_buf_);
        nvme_read_buf_ = nullptr;
        nvme_read_buf_size_ = 0;
    }
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

    for (int s = 0; s < (int)gpu_buf_.size(); s++) {
        if (gpu_buf_[s]) { cudaFree(gpu_buf_[s]); gpu_buf_[s] = nullptr; }
        if (s < (int)transfer_done_.size() && transfer_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(transfer_done_[s]));
            transfer_done_[s] = nullptr;
        }
        if (s < (int)compute_done_.size() && compute_done_[s]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(compute_done_[s]));
            compute_done_[s] = nullptr;
        }
    }
    gpu_buf_.clear();
    transfer_done_.clear();
    compute_done_.clear();
    current_layer_.clear();

    mmap_pinned_ = false;

    for (int s = 0; s < (int)staging_buf_.size(); s++) {
        if (staging_buf_[s]) {
            cudaFreeHost(staging_buf_[s]);
            staging_buf_[s] = nullptr;
        }
    }
    staging_buf_.clear();
    staging_ready_.clear();
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
    NT_CHECK(slot >= 0 && slot < n_slots_, "Slot index out of range");
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
// Get weight pointers for a RAM-cached layer (tier B) — zero-copy
// ============================================================
LayerWeightPtrs LayerStreamer::get_ram_weights(int layer_idx) const {
    int ram_idx = layer_idx - tier_config_.n_vram;
    NT_CHECK(tiered_mode_ && ram_idx >= 0 && ram_idx < (int)ram_cache_.size(),
             "get_ram_weights: layer not in RAM tier");

    const LayerLayout& lay = layers_[layer_idx];
    const uint8_t* ram_base = static_cast<const uint8_t*>(ram_cache_[ram_idx]);

    LayerWeightPtrs wp;
    wp.attn_q       = ram_base + lay.attn_q.gpu_offset;
    wp.attn_q_dtype = lay.attn_q.dtype;
    wp.attn_k       = ram_base + lay.attn_k.gpu_offset;
    wp.attn_k_dtype = lay.attn_k.dtype;
    wp.attn_v       = ram_base + lay.attn_v.gpu_offset;
    wp.attn_v_dtype = lay.attn_v.dtype;
    wp.attn_output  = ram_base + lay.attn_output.gpu_offset;
    wp.attn_o_dtype = lay.attn_output.dtype;
    wp.ffn_gate       = ram_base + lay.ffn_gate.gpu_offset;
    wp.ffn_gate_dtype = lay.ffn_gate.dtype;
    wp.ffn_up         = ram_base + lay.ffn_up.gpu_offset;
    wp.ffn_up_dtype   = lay.ffn_up.dtype;
    wp.ffn_down       = ram_base + lay.ffn_down.gpu_offset;
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

#ifdef USE_GPUNVME
        // NVMe BAR1 bulk read path (async — overlaps with GPU compute)
        if (bar1_enabled_ && nvme_vram_temp_) {
            const auto& nlay = nvme_layers_[layer_idx];
            gpunvme_err_t e = gpunvme_load_layer_vram(
                &nvme_loader_, nlay.start_lba, nlay.read_bytes,
                nvme_vram_temp_bar1_phys_);
            if (e == GPUNVME_OK) {
                current_layer_[slot] = layer_idx;
            }
        } else
#endif
        {
            memcpy_layer_to_staging(layer_idx, slot);
        }

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
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::VRAM) {
        // VRAM layers don't need staging
        return;
    }
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::RAM && !delta_mode_) {
        // RAM layers don't need staging (unless delta mode — then they use delta path)
        return;
    }

    if (mmap_pinned_) return;  // No staging needed — direct DMA path

    // Delta mode: copy U/V tensors from mmap'd .ntd to staging
    if (delta_mode_) {
        uint8_t* staging = static_cast<uint8_t*>(staging_buf_[slot]);
        const uint8_t* src = static_cast<const uint8_t*>(delta_mmap_);
        const auto& dlay = delta_layers_[layer_idx];
        for (int w = 0; w < 7; w++) {
            memcpy(staging + delta_slot_layout_.U_offset[w],
                   src + dlay.U[w].file_offset, dlay.U[w].nbytes);
            memcpy(staging + delta_slot_layout_.V_offset[w],
                   src + dlay.V[w].file_offset, dlay.V[w].nbytes);
        }
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            staging_ready_[slot] = true;
        }
        staging_ready_cv_.notify_all();
        return;
    }

#ifdef USE_GPUNVME
    if (nvme_initialized_) {
        const auto& nlay = nvme_layers_[layer_idx];

        // Tier 2 (BAR1): dispatch bulk NVMe read to worker thread (async)
        if (bar1_enabled_ && nvme_vram_temp_) {
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                staging_ready_[slot] = false;
                worker_request_ = {layer_idx, slot};
                worker_has_work_ = true;
            }
            worker_cv_.notify_one();
            return;  // Non-blocking — worker thread does the NVMe read
        }

        // Tier 1: Read the full file span into host pinned, then scatter-copy
        gpunvme_err_t err = gpunvme_load_layer(
            &nvme_loader_, nlay.start_lba, nlay.read_bytes, nvme_read_buf_);

        if (err == GPUNVME_OK) {
            // Scatter-copy each tensor from file layout to GPU layout in staging
            uint8_t* staging = static_cast<uint8_t*>(staging_buf_[slot]);
            const uint8_t* src = static_cast<const uint8_t*>(nvme_read_buf_);
            for (int t = 0; t < 7; t++) {
                memcpy(staging + nlay.tensors[t].gpu_offset,
                       src + nlay.tensors[t].read_offset,
                       nlay.tensors[t].nbytes);
            }

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
    NT_CHECK(slot >= 0 && slot < n_slots_, "Slot index out of range");

    // Map slot → transfer stream: alternate between the two DMA streams.
    // With 3 slots, slots 0 and 2 share STREAM_TRANSFER0; they are serialised
    // by their respective compute_done events so there is no ordering hazard.
    auto slot_xfer = [](int s) -> StreamType {
        return (s % 2 == 0) ? STREAM_TRANSFER0 : STREAM_TRANSFER1;
    };

    // Tier A: VRAM resident — no transfer needed, just record event
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::VRAM) {
        current_layer_[slot] = layer_idx;
        auto& dev = CUDADevice::instance();
        dev.record_event(transfer_done_[slot], slot_xfer(slot));
        return;
    }

#ifdef USE_GPUNVME
    // BAR1 Tier 2: wait for async NVMe read, then scatter 7 tensors to gpu_buf_[slot]
    if (bar1_enabled_ && nvme_vram_temp_ &&
        tiered_mode_ && layer_tier_[layer_idx] == LayerTier::NVME) {
        // Wait for worker thread to finish NVMe BAR1 read
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            staging_ready_cv_.wait(lock, [&] { return staging_ready_[slot]; });
        }

        if (current_layer_[slot] != layer_idx) {
            // NVMe read failed or wrong layer — fall through to other paths
            goto bar1_fallthrough;
        }

        const auto& nlay = nvme_layers_[layer_idx];
        uint8_t* dst = static_cast<uint8_t*>(gpu_buf_[slot]);
        const uint8_t* src = static_cast<const uint8_t*>(nvme_vram_temp_);

        auto& dev = CUDADevice::instance();
        StreamType xfer = (slot == 0) ? STREAM_TRANSFER0 : STREAM_TRANSFER1;
        cudaStream_t cs = static_cast<cudaStream_t>(dev.stream(xfer));

        dev.wait_event(xfer, compute_done_[slot]);

        for (int t = 0; t < 7; t++) {
            cudaMemcpyAsync(
                dst + nlay.tensors[t].gpu_offset,
                src + nlay.tensors[t].read_offset,
                nlay.tensors[t].nbytes,
                cudaMemcpyDeviceToDevice, cs);
        }

        // Sync: must complete before next prefetch_staging overwrites temp
        cudaStreamSynchronize(cs);
        dev.record_event(transfer_done_[slot], xfer);
        return;
    }
bar1_fallthrough:
#endif

    current_layer_[slot] = layer_idx;

    auto& dev = CUDADevice::instance();
    StreamType xfer = slot_xfer(slot);

    // Wait until compute on this slot is done (safe to overwrite GPU buffer)
    dev.wait_event(xfer, compute_done_[slot]);

    uint8_t* gpu_base = static_cast<uint8_t*>(gpu_buf_[slot]);

    // Tier B: async H2D from pinned RAM cache (per-tensor to handle requant)
    // In delta mode, tier B also uses the delta path (gpu_buf_ was resized to delta_buf_size_)
    if (tiered_mode_ && layer_tier_[layer_idx] == LayerTier::RAM && !delta_mode_) {
        int ram_idx = layer_idx - tier_config_.n_vram;
        uint8_t* ram_base = static_cast<uint8_t*>(ram_cache_[ram_idx]);
        const LayerLayout& lay = layers_[layer_idx];
        const TensorSlot* slots[] = {
            &lay.attn_q, &lay.attn_k, &lay.attn_v, &lay.attn_output,
            &lay.ffn_gate, &lay.ffn_up, &lay.ffn_down
        };
        for (int t = 0; t < 7; t++) {
            dev.memcpy_h2d_async(
                gpu_base + slots[t]->gpu_offset,
                ram_base + slots[t]->gpu_offset,
                slots[t]->xfer_nbytes, xfer);
        }
        dev.record_event(transfer_done_[slot], xfer);
        return;
    }

    // Tier C / legacy: from mmap or staging buffer
    if (delta_mode_ && mmap_pinned_) {
        // Delta mode: direct async copy of U/V tensors from pinned .ntd mmap
        const auto& dlay = delta_layers_[layer_idx];
        const uint8_t* src = static_cast<const uint8_t*>(delta_mmap_);
        for (int w = 0; w < 7; w++) {
            dev.memcpy_h2d_async(
                gpu_base + delta_slot_layout_.U_offset[w],
                src + dlay.U[w].file_offset, dlay.U[w].nbytes, xfer);
            dev.memcpy_h2d_async(
                gpu_base + delta_slot_layout_.V_offset[w],
                src + dlay.V[w].file_offset, dlay.V[w].nbytes, xfer);
        }
    } else if (mmap_pinned_) {
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
        size_t total = delta_mode_ ? delta_buf_size_ : layer_transfer_size(layer_idx);
        dev.memcpy_h2d_async(gpu_base, staging_buf_[slot], total, xfer);
    }

    dev.record_event(transfer_done_[slot], xfer);
}

} // namespace nt
