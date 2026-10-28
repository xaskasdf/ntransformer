// Expert offload for DeepSeek-V4.1 MXFP4 expert tensors.
//
// One CUDA device is used as an LRU cache for expert weight slices
// (gate/up/down, 6.27 MiB each in the real model) plus a small compute
// pipeline: GEMV (MXFP4) -> swiglu-clamp -> GEMV (down) -> weighted add.

#include "ds41_gpu.h"
#include "ds41_model.h"
#include "ds41_shared.h"
#include "ds41_attention.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <thread>

namespace nt {
namespace ds41 {

namespace {

constexpr int kThreads = 256;

MoeGpuStats moe_stats;

struct SharedLaunch {
    const void* gate;
    const void* up;
    const void* down;
    float* g;
    float* u;
    float* h;
    float* out;
    int ff;
    float limit;
};
bool prepare_shared(const SharedExpert& shared, int64_t hidden, SharedLaunch& launch);
void enqueue_shared(const SharedLaunch& launch, const float* x, int hidden, cudaStream_t stream);

// These helpers share the synchronous, non-reentrant expert-cache ownership.
// Reuse events only after the existing completion wait; no extra GPU wait is
// introduced. Failed/aborted traces never contribute partial phase durations.
struct MoeEvents {
    bool enabled = false;
    bool valid = true;
    cudaEvent_t events[6] = {};
    MoeEvents() {
        const char* flag = getenv("DS41_MOE_TRACE");
        enabled = flag && atoi(flag) != 0;
        if (enabled) for (auto& event : events) {
            if (cudaEventCreate(&event) != cudaSuccess) valid = false;
        }
    }
    ~MoeEvents() {
        for (auto event : events) if (event) cudaEventDestroy(event);
    }
};
class MoeTrace {
public:
    using Clock = std::chrono::steady_clock;
    explicit MoeTrace(cudaStream_t stream) : stream_(stream) {
        static MoeEvents events;
        events_ = &events;
        valid_ = events.valid;
        if (events.enabled) last_ = Clock::now();
    }
    ~MoeTrace() {
        if (events_->enabled && !finished_) ++moe_stats.trace_failures;
    }
    void event(int index) {
        if (events_->enabled && valid_ &&
            cudaEventRecord(events_->events[index], stream_) != cudaSuccess) valid_ = false;
    }
    void host(int index) {
        if (!events_->enabled) return;
        auto now = Clock::now();
        host_[index] = std::chrono::duration<double>(now - last_).count();
        last_ = now;
    }
    void finish() {
        if (!events_->enabled) return;
        double spans[5] = {};
        if (valid_) for (int i = 0; i < 5; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, events_->events[i], events_->events[i + 1]) != cudaSuccess) {
                valid_ = false;
                break;
            }
            spans[i] = ms * 0.001;
        }
        if (!valid_) return;
        for (int i = 0; i < 8; ++i) moe_stats.host_seconds[i] += host_[i];
        for (int i = 0; i < 5; ++i) moe_stats.stream_seconds[i] += spans[i];
        ++moe_stats.traced_calls;
        finished_ = true;
    }
private:
    MoeEvents* events_;
    cudaStream_t stream_;
    Clock::time_point last_;
    double host_[8] = {};
    bool valid_ = false;
    bool finished_ = false;
};

__constant__ float c_kvals[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

template<bool RegisterLut>
__device__ __forceinline__ float mxfp4_value(unsigned code) {
    if constexpr (!RegisterLut) return c_kvals[code];
    // The eight positive magnitudes fit in one register. Unlike a constant
    // memory lookup, different codes in a warp do not serialize table reads.
    const int magnitude = (0xC8643210u >> (4 * (code & 7))) & 15;
    return float((code & 8) ? -magnitude : magnitude); // both zero codes -> +0
}

bool register_lut_enabled() {
    static const bool enabled = [] {
        const char* value = getenv("DS41_MXFP4_REGISTER_LUT");
        return value && atoi(value) != 0;
    }();
    return enabled;
}

// 2^(e-128), matching kMXFP4Values being doubled (same as the CPU helper)
__device__ __forceinline__ float e8m0_half(uint8_t e) {
    uint32_t bits;
    if (e < 2) {
        bits = (e == 0) ? 0x00200000u : 0x00400000u;
    } else {
        bits = ((uint32_t)e) << 23;
    }
    float f;
    memcpy(&f, &bits, sizeof(float));
    return (e < 2) ? f : f * 0.5f;
}

// Keep the per-row operation order identical for serial and batched experts.
template<bool RegisterLut>
__device__ __forceinline__ float mxfp4_dot(const uint8_t* W, const float* x,
                                         int wid, int cols) {
    const int lane = threadIdx.x & 31;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * (size_t)nb * 17u;

    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 17u;
        const float d = e8m0_half(blk[0]);
        const uint8_t* qs = blk + 1;
        const float* xb = x + (size_t)b * 32u;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            acc += d * (mxfp4_value<RegisterLut>(qs[j] & 15) * xb[j] +
                        mxfp4_value<RegisterLut>(qs[j] >> 4) * xb[j + 16]);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    return acc;
}

// y[row] = W[row, :] @ x   (W: rows x cols, MXFP4 blocks of 32)
template<bool RegisterLut>
__global__ void k_gemv_mxfp4(const uint8_t* __restrict__ W,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int rows, int cols) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (wid >= rows) return;
    float acc = mxfp4_dot<RegisterLut>(W, x, wid, cols);
    if ((threadIdx.x & 31) == 0) y[wid] = acc;
}

struct ExpertBatch {
    const uint8_t* gate[8];
    const uint8_t* up[8];
    const uint8_t* down[8];
    float weights[8];
    int count;
};

// Expert pointers travel as launch arguments, without a metadata H2D copy.
// Each branch writes its own scratch; no cross-expert reduction happens here.
template<bool GateUp, bool RegisterLut>
__global__ void k_gemv_mxfp4_batch(ExpertBatch batch, const float* x,
                                  float* y, float* up, int rows, int cols) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (wid >= rows) return;
    const int expert = GateUp ? blockIdx.y / 2 : blockIdx.y;
    const uint8_t* w;
    if constexpr (GateUp) {
        const bool is_up = blockIdx.y & 1;
        w = is_up ? batch.up[expert] : batch.gate[expert];
        y = (is_up ? up : y) + (size_t)expert * rows;
    } else {
        w = batch.down[expert];
        x += (size_t)expert * cols;
        y += (size_t)expert * rows;
    }
    float acc = mxfp4_dot<RegisterLut>(w, x, wid, cols);
    if ((threadIdx.x & 31) == 0) y[wid] = acc;
}

void launch_mxfp4(const uint8_t* w, const float* x, float* y, int rows, int cols, cudaStream_t stream) {
    const int blocks = (rows * 32 + kThreads - 1) / kThreads;
    if (register_lut_enabled())
        k_gemv_mxfp4<true><<<blocks, kThreads, 0, stream>>>(w, x, y, rows, cols);
    else
        k_gemv_mxfp4<false><<<blocks, kThreads, 0, stream>>>(w, x, y, rows, cols);
}

template<bool GateUp>
void launch_mxfp4_batch(ExpertBatch batch, const float* x, float* y, float* up,
                        int rows, int cols, cudaStream_t stream) {
    dim3 grid((rows * 32 + kThreads - 1) / kThreads, GateUp ? 2 * batch.count : batch.count);
    if (register_lut_enabled())
        k_gemv_mxfp4_batch<GateUp, true><<<grid, kThreads, 0, stream>>>(batch, x, y, up, rows, cols);
    else
        k_gemv_mxfp4_batch<GateUp, false><<<grid, kThreads, 0, stream>>>(batch, x, y, up, rows, cols);
}

__global__ void k_expert_sum(ExpertBatch batch, const float* down, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    // Same ordered FP32 updates as the separate axpy launches.
#pragma unroll 1
    for (int k = 0; k < batch.count; ++k) acc += batch.weights[k] * down[(size_t)k * n + i];
    out[i] = acc;
}

// deepseek swiglu-clamp: h = min(g, lim)/(1+exp(-min(g,lim))) * clamp(u, -lim, lim)
__global__ void k_silu_clamp(const float* __restrict__ g,
                             const float* __restrict__ u,
                             float* __restrict__ h,
                             int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = fminf(g[i], limit);
    const float uv = fmaxf(-limit, fminf(u[i], limit));
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

__global__ void k_axpy(float* __restrict__ out, const float* __restrict__ y, float w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += w * y[i];
}

// ------------------------------------------------------------
// LRU expert cache on device
// ------------------------------------------------------------
struct CacheEntry {
    int slot;
    std::list<uint64_t>::iterator lru_it;
};

class ExpertCache {
public:
    bool ensure_device() {
        if (device_up_) return true;
        const char* disable = getenv("DS41_GPU");
        if (disable && atoi(disable) == 0) return false;
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
            fprintf(stderr, "ds41-gpu: no CUDA devices, offload disabled\n");
            return false;
        }
        if (cudaSetDevice(0) != cudaSuccess) return false;
        if (cudaStreamCreate(&stream_) != cudaSuccess) return false;
        device_up_ = true;
        return true;
    }

    bool ensure_init(size_t slice_bytes) {
        if (initialized_) return ok_;
        initialized_ = true;

        if (!ensure_device()) return false;

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);

        size_t cache_bytes = 40ull << 30;
        if (const char* gb = getenv("DS41_VRAM_GB")) {
            long v = atol(gb);
            if (v > 0) cache_bytes = (size_t)v << 30;
        }
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (cache_bytes > free_mem - (2ull << 30)) {
            cache_bytes = (free_mem > (4ull << 30)) ? (free_mem - (2ull << 30)) : 0;
        }

        slice_bytes_ = slice_bytes;
        n_slots_ = cache_bytes / slice_bytes_;
        if (n_slots_ < 64) {
            fprintf(stderr, "ds41-gpu: not enough VRAM for an expert cache\n");
            return false;
        }

        if (cudaMalloc(&cache_base_, n_slots_ * slice_bytes_) != cudaSuccess) {
            fprintf(stderr, "ds41-gpu: cache allocation failed (%zu slots)\n", n_slots_);
            return false;
        }

        auto alloc = [&](void** p, size_t bytes) {
            return cudaMalloc(p, bytes) == cudaSuccess;
        };
        // scratch buffers grow with the model; allocate generously for deepseek41
        if (!ensure_input(8192)) { return false; }
        // Independent expert branches retain FP32 intermediates until the
        // ordered final sum. Eight experts add 560 KiB over serial scratch.
        if (!alloc((void**)&gate_buf_, 8 * 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&up_buf_, 8 * 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&h_buf_, 8 * 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&down_buf_, 8 * 8192 * sizeof(float))) { return false; }
        if (!alloc((void**)&out_buf_, 8192 * sizeof(float))) { return false; }

        pin_n_ = 24;
        pin_ring_.resize(pin_n_);
        for (int i = 0; i < pin_n_; i++) {
            if (cudaMallocHost(&pin_ring_[i], slice_bytes_) != cudaSuccess) { return false; }
        }

        fprintf(stderr, "ds41-gpu: expert cache on %s: %zu slots x %.2f MiB = %.1f GiB\n",
                prop.name, n_slots_, slice_bytes_ / (1024.0 * 1024.0),
                (n_slots_ * slice_bytes_) / (1024.0 * 1024.0 * 1024.0));
        ok_ = true;
        return true;
    }

    bool available() const { return ok_; }

    void invalidate_entries() {
        map_.clear();
        lru_.clear();
        next_slot_ = 0;
    }

    // Attention can run before the first routed expert. Its input storage must
    // not depend on reserving the (potentially multi-GiB) expert weight cache.
    bool ensure_input(int64_t elements) {
        if (elements <= 0 || !ensure_device()) return false;
        if (x_capacity_ >= elements) return true;
        float* next = nullptr;
        if (cudaMalloc(&next, (size_t)elements * sizeof(float)) != cudaSuccess) return false;
        if (x_buf_) cudaFree(x_buf_);
        x_buf_ = next;
        x_capacity_ = elements;
        return true;
    }

    void reset_for_slice(size_t slice_bytes, size_t cache_bytes) {
        // not used; slot size fixed at first layer
        (void)slice_bytes; (void)cache_bytes;
    }

    // Looks a slice up; on a hit returns its device pointer, on a miss reserves a
    // slot (evicting when needed) and returns it without copying.
    void* lookup_or_reserve(uint64_t key, bool* hit) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            hits_++;
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
            *hit = true;
            return slot_ptr(it->second.slot);
        }
        *hit = false;
        misses_++;
        int slot;
        if (next_slot_ < n_slots_) {
            slot = next_slot_++;
        } else {
            uint64_t victim_key = lru_.back();
            lru_.pop_back();
            auto vit = map_.find(victim_key);
            slot = vit->second.slot;
            map_.erase(vit);
            evictions_++;
        }
        lru_.push_front(key);
        map_[key] = CacheEntry{slot, lru_.begin()};
        return slot_ptr(slot);
    }

    // pinned staging buffers (pageable cudaMemcpyAsync runs at <1 GB/s here)
    char* pin_ring(int i) const { return pin_ring_[i % pin_n_]; }
    int pin_ring_size() const { return pin_n_; }
    size_t slice_bytes() const { return slice_bytes_; }

    unsigned long long hits() const { return hits_; }
    unsigned long long misses() const { return misses_; }
    unsigned long long evictions() const { return evictions_; }

    cudaStream_t stream() const { return stream_; }
    float* x_buf() const { return x_buf_; }
    float* gate_buf() const { return gate_buf_; }
    float* up_buf() const { return up_buf_; }
    float* h_buf() const { return h_buf_; }
    float* down_buf() const { return down_buf_; }
    float* out_buf() const { return out_buf_; }

private:
    void* slot_ptr(int i) const {
        return static_cast<char*>(cache_base_) + (size_t)i * slice_bytes_;
    }

    bool initialized_ = false;
    bool device_up_ = false;
    bool ok_ = false;
    cudaStream_t stream_ = nullptr;
    void* cache_base_ = nullptr;
    size_t slice_bytes_ = 0;
    size_t n_slots_ = 0;
    size_t next_slot_ = 0;

    std::list<uint64_t> lru_;
    std::unordered_map<uint64_t, CacheEntry> map_;

    unsigned long long hits_ = 0, misses_ = 0, evictions_ = 0;

    float* x_buf_ = nullptr;
    int64_t x_capacity_ = 0;
    float* gate_buf_ = nullptr;
    float* up_buf_ = nullptr;
    float* h_buf_ = nullptr;
    float* down_buf_ = nullptr;
    float* out_buf_ = nullptr;
    int pin_n_ = 0;
    std::vector<char*> pin_ring_;
};

ExpertCache& cache() {
    static ExpertCache c;
    return c;
}

inline uint64_t make_key(const void* base, int64_t expert) {
    uint64_t p = reinterpret_cast<uint64_t>(base);
    return p * 1000003ull + (uint64_t)expert * 2654435761ull;
}

} // namespace

bool gpu_experts_enabled() {
    const char* disable = getenv("DS41_GPU");
    if (disable && atoi(disable) == 0) return false;
    int ndev = 0;
    return cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
}

MoeGpuStats gpu_moe_stats() { return moe_stats; }

// pinned staging for host<->device copies (pageable copies run at <1 GB/s)
char* pin_stage(size_t bytes) {
    static char* p = nullptr;
    static size_t cap = 0;
    if (bytes > cap) {
        if (p) cudaFreeHost(p);
        if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
            p = nullptr;
            cap = 0;
            return nullptr;
        }
        cap = bytes;
    }
    return p;
}

bool h2d_stage(void* dev, const void* host, size_t bytes, cudaStream_t s) {
    char* p = pin_stage(bytes);
    if (!p) return false;
    memcpy(p, host, bytes);
    return cudaMemcpyAsync(dev, p, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess;
}

bool d2h_stage(void* host, const void* dev, size_t bytes, cudaStream_t s) {
    char* p = pin_stage(bytes);
    if (!p) return false;
    if (cudaMemcpyAsync(p, dev, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) return false;
    if (cudaStreamSynchronize(s) != cudaSuccess) return false;
    memcpy(host, p, bytes);
    return true;
}

// static rope table buffers
float* rope_dev_cos() {
    static float* p = nullptr;
    if (!p && cudaMalloc(&p, 4096 * 4) != cudaSuccess) return nullptr;
    return p;
}
float* rope_dev_sin() {
    static float* p = nullptr;
    if (!p && cudaMalloc(&p, 4096 * 4) != cudaSuccess) return nullptr;
    return p;
}
bool gpu_moe_experts(const float* x, int64_t hidden,
                     const void* gate_base, const void* up_base, const void* down_base,
                     int64_t ff,
                     const int* experts, const float* weights, int n_used,
                     float limit, float* out) {
    return gpu_moe_experts(x, hidden, gate_base, up_base, down_base, ff,
                           experts, weights, n_used, limit, out, {});
}

static bool gpu_moe_impl(const float* x, int64_t hidden,
                     const void* gate_base, const void* up_base, const void* down_base,
                     int64_t ff, const int* experts, const float* weights, int n_used,
                     float limit, float* out, const std::function<void()>& host_work,
                     const SharedExpert* shared, float* shared_out) {
    // The existing cache scratch has these capacities. Reject unsupported
    // dimensions before publishing cache entries or queuing any work.
    if (!x || !out || !gate_base || !up_base || !down_base || !experts || !weights ||
        hidden <= 0 || hidden > 8192 || hidden % 32 || ff <= 0 || ff > 4096 || ff % 32 ||
        n_used <= 0 || n_used > 8) return false;
    for (int k = 0; k < n_used; ++k) if (experts[k] < 0) return false;
    ExpertCache& c = cache();
    const size_t row_bytes_gate = (size_t)(hidden / 32) * 17;
    const size_t row_bytes_down = (size_t)(ff / 32) * 17;
    const size_t gate_slice = (size_t)ff * row_bytes_gate;
    const size_t down_slice = (size_t)hidden * row_bytes_down;
    const size_t slice_bytes = gate_slice > down_slice ? gate_slice : down_slice;

    if (!c.ensure_init(slice_bytes) || !c.available()) return false;
    if (c.slice_bytes() != slice_bytes) return false;

    cudaStream_t s = c.stream();
    static const bool batched = [] {
        const char* flag = getenv("DS41_MOE_BATCH");
        return !flag || atoi(flag) != 0;
    }();
    static const bool pooled_staging = [] {
        const char* flag = getenv("DS41_STAGING_POOL");
        return flag && atoi(flag) != 0;
    }();
    static const bool shared_early = [] {
        const char* flag = getenv("DS41_SHARED_EARLY");
        return !flag || atoi(flag) != 0;
    }();
    SharedLaunch shared_launch{};
    if (shared && (!shared_out || !prepare_shared(*shared, hidden, shared_launch))) return false;
    // Reserve readback storage before submitting work. This pinned buffer is
    // owned by this synchronous call until the completion wait below.
    char* readback = pin_stage((size_t)hidden * sizeof(float) * (shared ? 2 : 1));
    if (!readback) return false;
    auto fail = [&]() {
        cudaStreamSynchronize(s);
        c.invalidate_entries(); // never reuse a slice whose upload failed
        return false;
    };

    MoeTrace trace(s);
    trace.event(0);

    if (cudaMemcpyAsync(c.x_buf(), x, hidden * sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) return fail();
    // The batched reduction writes every output from its own +0 accumulator.
    if (!batched && cudaMemsetAsync(c.out_buf(), 0, hidden * sizeof(float), s) != cudaSuccess) return fail();
    trace.event(1);
    trace.host(0);
    // The resident shared expert depends only on x, not on routing or misses.
    // Submit it before host staging so the device can work during that interval.
    if (shared && shared_early) enqueue_shared(shared_launch, c.x_buf(), (int)hidden, s);

    struct Miss { size_t idx; const void* host; void* dst; size_t bytes; };
    std::vector<Miss> misses;
    misses.reserve(3 * n_used);
    const void* dev_gate[8];
    const void* dev_up[8];
    const void* dev_down[8];

    for (int k = 0; k < n_used; k++) {
        const int e = experts[k];
        const uint8_t* hg = static_cast<const uint8_t*>(gate_base) + (size_t)e * gate_slice;
        const uint8_t* hu = static_cast<const uint8_t*>(up_base)   + (size_t)e * gate_slice;
        const uint8_t* hd = static_cast<const uint8_t*>(down_base) + (size_t)e * down_slice;
        bool hit;
        void* d;
        d = c.lookup_or_reserve(make_key(gate_base, e), &hit); dev_gate[k] = d;
        if (!hit) misses.push_back({(size_t)k, hg, d, gate_slice});
        d = c.lookup_or_reserve(make_key(up_base, e),   &hit); dev_up[k] = d;
        if (!hit) misses.push_back({(size_t)(100 + k), hu, d, gate_slice});
        d = c.lookup_or_reserve(make_key(down_base, e), &hit); dev_down[k] = d;
        if (!hit) misses.push_back({(size_t)(200 + k), hd, d, down_slice});
    }
    trace.host(1);

    // stage misses into pinned memory in parallel, then enqueue async H2D copies
    if (!misses.empty()) {
        const int npin = c.pin_ring_size();
        if (misses.size() > (size_t)npin) return fail();
        auto copy = [&](int64_t i) {
            memcpy(c.pin_ring((int)i), misses[i].host, misses[i].bytes);
        };
        try {
            if (pooled_staging) {
                parallel_for((int64_t)misses.size(), 0, [&](int64_t i, int) { copy(i); });
            } else {
                // Retained as a measurement control. jthread also joins if
                // worker creation throws after some workers have started.
                std::vector<std::jthread> copiers;
                copiers.reserve(misses.size());
                for (size_t i = 0; i < misses.size(); ++i) copiers.emplace_back(copy, i);
            }
        } catch (...) {
            fail();
            throw;
        }
    }
    trace.host(2);
    trace.event(2);
    uint64_t uploaded_bytes = 0;
    if (!misses.empty()) {
        for (size_t i = 0; i < misses.size(); i++) {
            if (cudaMemcpyAsync(misses[i].dst, c.pin_ring((int)i), misses[i].bytes,
                                cudaMemcpyHostToDevice, s) != cudaSuccess) return fail();
            uploaded_bytes += misses[i].bytes;
        }
    }
    trace.event(3);
    trace.host(3);

    if (shared && !shared_early) enqueue_shared(shared_launch, c.x_buf(), (int)hidden, s);
    if (batched) {
        ExpertBatch batch{};
        batch.count = n_used;
        for (int k = 0; k < n_used; ++k) {
            batch.gate[k] = static_cast<const uint8_t*>(dev_gate[k]);
            batch.up[k] = static_cast<const uint8_t*>(dev_up[k]);
            batch.down[k] = static_cast<const uint8_t*>(dev_down[k]);
            batch.weights[k] = weights[k];
        }
        launch_mxfp4_batch<true>(
            batch, c.x_buf(), c.gate_buf(), c.up_buf(), (int)ff, (int)hidden, s);
        k_silu_clamp<<<((int)ff * n_used + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            c.gate_buf(), c.up_buf(), c.h_buf(), (int)ff * n_used, limit);
        launch_mxfp4_batch<false>(
            batch, c.h_buf(), c.down_buf(), nullptr, (int)hidden, (int)ff, s);
        k_expert_sum<<<((int)hidden + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            batch, c.down_buf(), c.out_buf(), (int)hidden);
    } else for (int k = 0; k < n_used; k++) {
        const int rows_gate = (int)ff;
        const int cols_gate = (int)hidden;
        launch_mxfp4(static_cast<const uint8_t*>(dev_gate[k]), c.x_buf(), c.gate_buf(), rows_gate, cols_gate, s);
        launch_mxfp4(static_cast<const uint8_t*>(dev_up[k]), c.x_buf(), c.up_buf(), rows_gate, cols_gate, s);

        k_silu_clamp<<<(rows_gate + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            c.gate_buf(), c.up_buf(), c.h_buf(), rows_gate, limit);

        const int rows_down = (int)hidden;
        const int cols_down = (int)ff;
        launch_mxfp4(static_cast<const uint8_t*>(dev_down[k]), c.h_buf(), c.down_buf(), rows_down, cols_down, s);

        k_axpy<<<(rows_down + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            c.out_buf(), c.down_buf(), weights[k], rows_down);
    }
    trace.event(4);

    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(readback, c.out_buf(), hidden * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess) {
        return fail();
    }
    if (shared && cudaMemcpyAsync(readback + hidden * sizeof(float), shared_launch.out,
            hidden * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess) return fail();
    trace.event(5);
    trace.host(4);
    try {
        if (host_work) host_work();
    } catch (...) {
        fail();
        throw;
    }
    trace.host(5);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        fprintf(stderr, "ds41-gpu: moe output copy failed\n");
        return fail();
    }
    trace.host(6);
    memcpy(out, readback, hidden * sizeof(float));
    if (shared) memcpy(shared_out, readback + hidden * sizeof(float), hidden * sizeof(float));
    trace.host(7);
    trace.finish();
    ++moe_stats.calls;
    moe_stats.batched_calls += batched;
    moe_stats.kernel_launches += batched ? 4 : 5 * n_used;
    moe_stats.uploaded_bytes += uploaded_bytes;
    moe_stats.staged_slices += misses.size();
    moe_stats.staging_pool_calls += pooled_staging && !misses.empty();
    moe_stats.transient_copy_threads += pooled_staging ? 0 : misses.size();
    moe_stats.register_lut_launches += register_lut_enabled() ? (batched ? 2 : 3 * n_used) : 0;
    if (shared) {
        ++moe_stats.shared_calls;
        moe_stats.shared_early_calls += shared_early;
        moe_stats.shared_kernel_launches += 3;
    }
    return true;
}

bool gpu_moe_experts(const float* x, int64_t hidden,
                     const void* gate, const void* up, const void* down,
                     int64_t ff, const int* experts, const float* weights, int n_used,
                     float limit, float* out, const std::function<void()>& host_work) {
    return gpu_moe_impl(x, hidden, gate, up, down, ff, experts, weights, n_used,
                        limit, out, host_work, nullptr, nullptr);
}

bool gpu_moe_experts_shared(const float* x, int64_t hidden,
                            const void* gate, const void* up, const void* down, int64_t ff,
                            const int* experts, const float* weights, int n_used,
                            float limit, float* out, const SharedExpert& shared, float* shared_out) {
    return gpu_moe_impl(x, hidden, gate, up, down, ff, experts, weights, n_used,
                        limit, out, {}, &shared, shared_out);
}
bool gpu_gemv_expert(const void* base, int64_t expert, int64_t rows, int64_t cols,
                     const float* x, float* y) {
    ExpertCache& c = cache();
    const size_t row_bytes = (size_t)(cols / 32) * 17;
    const size_t slice = (size_t)rows * row_bytes;
    if (!c.ensure_init(slice) || !c.available()) return false;
    cudaStream_t s = c.stream();
    const uint8_t* host = static_cast<const uint8_t*>(base) + (size_t)expert * slice;
    bool hit = false;
    void* dev = c.lookup_or_reserve(make_key(base, expert), &hit);
    if (!hit) {
        memcpy(c.pin_ring(0), host, slice);
        cudaMemcpyAsync(dev, c.pin_ring(0), slice, cudaMemcpyHostToDevice, s);
    }
    h2d_stage(c.x_buf(), x, (size_t)cols * sizeof(float), s);
    launch_mxfp4(static_cast<const uint8_t*>(dev), c.x_buf(), c.down_buf(), (int)rows, (int)cols, s);
    return d2h_stage(y, c.down_buf(), (size_t)rows * sizeof(float), s);
}

__device__ __forceinline__ float bf16_to_f32_dev(uint16_t h) {
    uint32_t f = ((uint32_t)h) << 16;
    float r;
    memcpy(&r, &f, sizeof(float));
    return r;
}

// BF16 rows x cols gemv, warp per row
__global__ void k_gemv_bf16(const uint16_t* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const uint16_t* row = W + (size_t)wid * (size_t)cols;
    float acc = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        acc += bf16_to_f32_dev(row[i]) * x[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}

struct LmHead {
    void* weight = nullptr;
    int64_t rows = 0, cols = 0;
    float* x_buf = nullptr;
    float* y_buf = nullptr;
};
LmHead g_lm;

bool gpu_lm_head_init(const void* weight, int64_t rows, int64_t cols) {
    ExpertCache& c = cache();
    if (!c.ensure_device()) return false;
    if (g_lm.weight && g_lm.rows == rows && g_lm.cols == cols) return true;
    if (g_lm.weight) {
        cudaFree(g_lm.weight);
        cudaFree(g_lm.x_buf);
        cudaFree(g_lm.y_buf);
        g_lm = LmHead{};
    }
    const size_t bytes = (size_t)rows * (size_t)cols * 2;
    if (cudaMalloc(&g_lm.weight, bytes) != cudaSuccess) {
        fprintf(stderr, "ds41-gpu: lm head allocation failed (%.2f GiB)\n", bytes / 1073741824.0);
        return false;
    }
    cudaMemcpyAsync(g_lm.weight, weight, bytes, cudaMemcpyHostToDevice, c.stream());
    if (cudaMalloc(&g_lm.x_buf, (size_t)cols * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&g_lm.y_buf, (size_t)rows * sizeof(float)) != cudaSuccess) return false;
    if (cudaStreamSynchronize(c.stream()) != cudaSuccess) return false;
    g_lm.rows = rows;
    g_lm.cols = cols;
    fprintf(stderr, "ds41-gpu: lm head resident (%.2f GiB)\n", bytes / 1073741824.0);
    return true;
}

bool gpu_lm_head(const float* x, float* logits, int64_t rows, int64_t cols) {
    if (!g_lm.weight || g_lm.rows != rows || g_lm.cols != cols) return false;
    ExpertCache& c = cache();
    cudaStream_t s = c.stream();
    h2d_stage(g_lm.x_buf, x, (size_t)cols * sizeof(float), s);
    const int threads = (int)rows * 32;
    k_gemv_bf16<<<(threads + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        static_cast<const uint16_t*>(g_lm.weight), g_lm.x_buf, g_lm.y_buf, (int)rows, (int)cols);
    return d2h_stage(logits, g_lm.y_buf, (size_t)rows * sizeof(float), s);
}

// Q8_0 gemv, warp per row: y[row] = W[row,:] @ x
template<bool Grouped = false>
__global__ void k_gemv_q8_0(const uint8_t* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    if constexpr (Grouped) {
        // Independent matrices occupy grid.y; each row keeps the same warp,
        // Q8 block traversal and reduction as the ordinary GEMV.
        W += (size_t)blockIdx.y * rows * nb * 34u;
        x += (size_t)blockIdx.y * cols;
        y += (size_t)blockIdx.y * rows;
    }
    const uint8_t* row = W + (size_t)wid * (size_t)nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh; memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = x + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}

// y = x * rsqrt(mean(x^2)+eps) * w   (single block)
__global__ void k_rms_norm_w(float* __restrict__ x, const float* __restrict__ w,
                             int n, float eps) {
    extern __shared__ float sh[];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(sh[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = w ? x[i] * scale * w[i] : x[i] * scale;
    }
}

// NORM-type rope (consecutive pairs) at [offset, offset+dims) of each head
__global__ void k_rope_norm_k(float* __restrict__ v, int n_heads, int head_dim,
                              int offset, int dims,
                              const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                              int backward) {
    const int half = dims / 2;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_heads * half) return;
    const int h = p / half;
    const int q = p % half;
    float* vh = v + (size_t)h * head_dim + offset;
    const int i0 = 2 * q;
    const float x0 = vh[i0];
    const float x1 = vh[i0 + 1];
    const float c = cos_t[q];
    const float s = sin_t[q] * (backward ? -1.0f : 1.0f);
    vh[i0]     = x0 * c - x1 * s;
    vh[i0 + 1] = x0 * s + x1 * c;
}

// resident tensor registry (attention backbone weights, uploaded once)
class ResidentRegistry {
public:
    const void* get(const void* host, size_t bytes) {
        if (!host) return nullptr;
        auto it = map_.find(host);
        if (it != map_.end()) return it->second;
        void* dev = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) return nullptr;
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, cache().stream()) != cudaSuccess) {
            cudaFree(dev);
            return nullptr;
        }
        if (cudaStreamSynchronize(cache().stream()) != cudaSuccess) {
            cudaFree(dev);
            return nullptr;
        }
        map_[host] = dev;
        bytes_ += bytes;
        return dev;
    }
    size_t bytes() const { return bytes_; }
private:
    std::unordered_map<const void*, void*> map_;
    size_t bytes_ = 0;
};
ResidentRegistry g_resident;

namespace {
ResidentRegistry shared_resident;
float* shared_scratch = nullptr;
size_t shared_capacity = 0;

bool prepare_shared(const SharedExpert& shared, int64_t hidden, SharedLaunch& launch) {
    if (!shared.gate || !shared.up || !shared.down || hidden <= 0 || hidden > 8192 ||
        hidden % 32 || shared.ff <= 0 || shared.ff > 8192 || shared.ff % 32 ||
        !cache().ensure_input(hidden)) return false;
    const size_t bytes = (size_t)hidden * (shared.ff / 32) * 34;
    const void* gate = shared_resident.get(shared.gate, bytes);
    const void* up = shared_resident.get(shared.up, bytes);
    const void* down = shared_resident.get(shared.down, bytes);
    moe_stats.shared_weight_bytes = shared_resident.bytes();
    if (!gate || !up || !down) return false;
    const size_t need = (size_t)shared.ff * 3 + hidden;
    if (need > shared_capacity) {
        float* next = nullptr;
        if (cudaMalloc(&next, need * sizeof(float)) != cudaSuccess) return false;
        if (shared_scratch) cudaFree(shared_scratch);
        shared_scratch = next;
        shared_capacity = need;
    }
    launch = {gate, up, down, shared_scratch, shared_scratch + shared.ff,
              shared_scratch + 2 * shared.ff, shared_scratch + 3 * shared.ff,
              (int)shared.ff, shared.limit};
    return true;
}

void enqueue_shared(const SharedLaunch& launch, const float* x, int hidden, cudaStream_t stream) {
    detail::launch_shared_q8(launch.gate, launch.up, x, launch.g, launch.u, launch.ff, hidden, true, stream);
    detail::launch_shared_activation(launch.g, launch.u, launch.h, launch.ff, launch.limit, stream);
    detail::launch_shared_q8(launch.down, nullptr, launch.h, launch.out, nullptr, hidden, launch.ff, true, stream);
}
} // namespace

bool gpu_shared_expert(const float* x, int64_t hidden, const SharedExpert& shared, float* out) {
    if (!x || !out) return false;
    SharedLaunch launch{};
    if (!prepare_shared(shared, hidden, launch)) return false;
    auto& c = cache();
    auto fail = [&] { cudaStreamSynchronize(c.stream()); return false; };
    if (!h2d_stage(c.x_buf(), x, hidden * sizeof(float), c.stream())) return fail();
    enqueue_shared(launch, c.x_buf(), (int)hidden, c.stream());
    if (cudaGetLastError() != cudaSuccess || !d2h_stage(out, launch.out, hidden * sizeof(float), c.stream())) return fail();
    return true;
}

// attention scratch, grown on demand
struct AttnScratch {
    float* qr = nullptr;
    float* q = nullptr;
    float* kv = nullptr;
    float* oa = nullptr;
    float* out = nullptr;
    int64_t qr_n = 0, q_n = 0, kv_n = 0, oa_n = 0, out_n = 0;
};
AttnScratch g_attn;
static BackboneStats g_backbone_stats;

BackboneStats gpu_backbone_stats() {
    return g_backbone_stats;
}

float* scratch_slot(float*& buf, int64_t& have, int64_t need) {
    if (have >= need && buf) return buf;
    if (buf) cudaFree(buf);
    buf = nullptr;
    if (cudaMalloc(&buf, (size_t)need * sizeof(float)) != cudaSuccess) {
        buf = nullptr;
        have = 0;
        return nullptr;
    }
    have = need;
    return buf;
}

size_t q8_matrix_bytes(int64_t rows, int64_t cols) {
    return (size_t)rows * (size_t)(cols / 32) * 34u;
}

bool gpu_backbone_enabled() {
    return cache().ensure_device();
}

struct GpuAttentionState {
    int layers, window, heads, dim, rope_dim = 0;
    int query_layer = -1;
    int64_t query_position = -1;
    float *raw = nullptr, *query = nullptr, *output = nullptr, *cos_t = nullptr, *sin_t = nullptr, *sinks = nullptr;
    float* scores = nullptr;
    int64_t scores_capacity = 0;
    int* selected = nullptr;
    int selected_capacity = 0;
    char* staging = nullptr;
    size_t staging_capacity = 0;
    struct Compressed { float* data = nullptr; int capacity = 0, uploaded = 0; };
    std::vector<Compressed> compressed;
    std::vector<int64_t> next_position;
    GpuAttentionState(int l, int w, int h, int d) : layers(l), window(w), heads(h), dim(d), compressed(l), next_position(l, 0) {}
    ~GpuAttentionState() {
        // All public calls complete their submissions before returning.
        for (auto& c : compressed) if (c.data) cudaFree(c.data);
        for (float* p : {raw, query, output, cos_t, sin_t, sinks, scores}) if (p) cudaFree(p);
        if (selected) cudaFree(selected);
        if (staging) cudaFreeHost(staging);
    }
    bool stage(size_t bytes) {
        if (bytes <= staging_capacity) return true;
        char* replacement = nullptr;
        if (cudaHostAlloc(&replacement, bytes, cudaHostAllocDefault) != cudaSuccess) return false;
        if (staging) cudaFreeHost(staging);
        staging = replacement; staging_capacity = bytes;
        return true;
    }
};

GpuAttentionState* gpu_attention_create(int layers, int window, int heads, int dim) {
    if (layers < 1 || layers > 256 || window < 1 || window > 4096 || heads < 1 || heads > 256 || dim < 2 || dim > 512 ||
        !cache().ensure_device()) return nullptr;
    auto s = std::make_unique<GpuAttentionState>(layers, window, heads, dim);
    if (cudaMalloc(&s->raw, (size_t)layers * window * dim * 4) != cudaSuccess ||
        cudaMalloc(&s->query, (size_t)heads * dim * 4) != cudaSuccess ||
        cudaMalloc(&s->output, (size_t)heads * dim * 4) != cudaSuccess ||
        cudaMalloc(&s->cos_t, (size_t)dim * 4) != cudaSuccess ||
        cudaMalloc(&s->sin_t, (size_t)dim * 4) != cudaSuccess ||
        cudaMalloc(&s->sinks, (size_t)heads * 4) != cudaSuccess) return nullptr;
    return s.release();
}

void gpu_attention_destroy(GpuAttentionState* state) { delete state; }

void gpu_attention_reset(GpuAttentionState* state) {
    if (!state) return;
    std::fill(state->next_position.begin(), state->next_position.end(), 0);
    for (auto& c : state->compressed) c.uploaded = 0;
    state->query_layer = -1; state->query_position = -1;
}

bool gpu_attention_ready(const GpuAttentionState* state, int layer, int64_t position) {
    return state && layer >= 0 && layer < state->layers && position >= 0 && position < INT32_MAX &&
           state->next_position[layer] == position;
}

bool gpu_attention_read_query(GpuAttentionState* state, float* query) {
    return state && query && state->query_layer >= 0 &&
           d2h_stage(query, state->query, (size_t)state->heads * state->dim * 4, cache().stream());
}

// Reserves all buffers before queueing copies. The per-state pinned staging
// allocation is separate from projection readback scratch and stays alive
// through the tail/probe completion wait.
static bool enqueue_attention(GpuAttentionState* s, int layer, int64_t position,
                               const AttentionRows& rows, const float* sinks,
                               size_t& comp_bytes, size_t& metadata_bytes) {
    if (!s || !sinks || layer != s->query_layer || position != s->query_position ||
        rows.count < 0 || rows.count > 32768 || rows.n_selected < 0 ||
        rows.n_selected + std::min<int64_t>(s->window, position + 1) > 8192 ||
        (rows.count && (!rows.rows || rows.source < 0 || rows.source >= s->layers)) ||
        (rows.n_selected && (!rows.selected || !rows.count))) return false;
    for (int i = 0; i < rows.n_selected; ++i)
        if (rows.selected[i] < 0 || rows.selected[i] >= rows.count) return false;
    auto stream = cache().stream();
    const int n_keys = (int)std::min<int64_t>(s->window, position + 1) + rows.n_selected;
    if (!scratch_slot(s->scores, s->scores_capacity, (int64_t)s->heads * n_keys)) return false;
    if (rows.n_selected > s->selected_capacity) {
        int* replacement = nullptr;
        if (cudaMalloc(&replacement, (size_t)rows.n_selected * 4) != cudaSuccess) return false;
        if (s->selected) cudaFree(s->selected);
        s->selected = replacement; s->selected_capacity = rows.n_selected;
    }
    GpuAttentionState::Compressed* comp = rows.count ? &s->compressed[rows.source] : nullptr;
    if (comp && comp->capacity < rows.count) {
        int capacity = std::max(16, comp->capacity);
        while (capacity < rows.count) capacity *= 2;
        float* replacement = nullptr;
        if (cudaMalloc(&replacement, (size_t)capacity * s->dim * 4) != cudaSuccess) return false;
        if (comp->data) cudaFree(comp->data);
        comp->data = replacement; comp->capacity = capacity; comp->uploaded = 0;
    }
    const int old_rows = comp ? comp->uploaded : 0;
    comp_bytes = comp && rows.count > old_rows ? (size_t)(rows.count - old_rows) * s->dim * 4 : 0;
    const size_t indices_bytes = (size_t)rows.n_selected * 4;
    metadata_bytes = indices_bytes + (size_t)s->heads * 4;
    if (!s->stage(comp_bytes + metadata_bytes)) return false;
    if (comp_bytes) memcpy(s->staging, rows.rows + (size_t)old_rows * s->dim, comp_bytes);
    if (indices_bytes) memcpy(s->staging + comp_bytes, rows.selected, indices_bytes);
    memcpy(s->staging + comp_bytes + indices_bytes, sinks, (size_t)s->heads * 4);
    auto fail = [&] { cudaStreamSynchronize(stream); if (comp) comp->uploaded = 0; return false; };
    if ((comp_bytes && cudaMemcpyAsync(comp->data + (size_t)old_rows * s->dim, s->staging, comp_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) ||
        (indices_bytes && cudaMemcpyAsync(s->selected, s->staging + comp_bytes, indices_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) ||
        cudaMemcpyAsync(s->sinks, s->staging + comp_bytes + indices_bytes, (size_t)s->heads * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail();
    detail::launch_attention(s->query, s->raw + (size_t)layer * s->window * s->dim, s->window, position,
        comp ? comp->data : nullptr, s->selected, rows.n_selected, s->sinks, s->cos_t, s->sin_t,
        s->heads, s->dim, s->rope_dim, 1.0f / std::sqrt((float)s->dim), s->scores, s->output, stream);
    if (cudaGetLastError() != cudaSuccess) return fail();
    if (comp) comp->uploaded = std::max(comp->uploaded, rows.count);
    return true;
}

bool gpu_attention_probe(GpuAttentionState* s, int layer, int64_t position,
                         const float* query, const float* kv, const float* cos_t,
                         const float* sin_t, int rope_dim, const AttentionRows& rows,
                         const float* sinks, float* out) {
    if (!gpu_attention_ready(s, layer, position) || !query || !kv || !out || !sinks || !cos_t || !sin_t ||
        rope_dim < 0 || rope_dim > s->dim || rope_dim % 2) return false;
    const size_t qbytes = (size_t)s->heads * s->dim * 4, kvbytes = (size_t)s->dim * 4, rbytes = (size_t)rope_dim / 2 * 4;
    if (!s->stage(qbytes + kvbytes + 2 * rbytes)) return false;
    memcpy(s->staging, query, qbytes); memcpy(s->staging + qbytes, kv, kvbytes);
    memcpy(s->staging + qbytes + kvbytes, cos_t, rbytes); memcpy(s->staging + qbytes + kvbytes + rbytes, sin_t, rbytes);
    auto stream = cache().stream();
    auto fail = [&] { cudaStreamSynchronize(stream); return false; };
    if (cudaMemcpyAsync(s->query, s->staging, qbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->raw + ((size_t)layer * s->window + position % s->window) * s->dim, s->staging + qbytes, kvbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->cos_t, s->staging + qbytes + kvbytes, rbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->sin_t, s->staging + qbytes + kvbytes + rbytes, rbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return fail();
    s->query_layer = layer; s->query_position = position; s->next_position[layer] = position + 1; s->rope_dim = rope_dim;
    size_t comp_bytes = 0, metadata_bytes = 0;
    if (!enqueue_attention(s, layer, position, rows, sinks, comp_bytes, metadata_bytes) ||
        !d2h_stage(out, s->output, qbytes, stream)) return fail();
    return true;
}

static bool gpu_dense_projection(const void* w, const float* x, float* out, int64_t rows, int64_t cols,
                                 bool bf16, bool avx2_order) {
    if (!w || !x || !out || rows < 1 || rows > 4096 || cols < 1 || cols > 65536) return false;
    ExpertCache& c = cache();
    if (!c.ensure_input(cols)) return false;
    const void* device_w = g_resident.get(w, (size_t)rows * cols * (bf16 ? 2 : 4));
    if (!device_w) return false;
    static float* device_out = nullptr;
    static int64_t capacity = 0;
    float* y = scratch_slot(device_out, capacity, rows);
    char* readback = pin_stage((size_t)std::max(rows, cols) * sizeof(float));
    if (!y || !readback) return false;
    cudaStream_t s = c.stream();
    auto fail = [&]() { cudaStreamSynchronize(s); return false; };
    if (!h2d_stage(c.x_buf(), x, (size_t)cols * sizeof(float), s)) return fail();
    detail::launch_router(device_w, c.x_buf(), y, (int)rows, (int)cols, bf16, avx2_order, s);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(readback, y, (size_t)rows * sizeof(float), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) return fail();
    memcpy(out, readback, (size_t)rows * sizeof(float));
    return true;
}

bool gpu_router(const void* w, const float* x, float* out, int64_t rows, int64_t cols,
                 bool bf16, bool avx2_order) {
    if (!gpu_dense_projection(w, x, out, rows, cols, bf16, avx2_order)) return false;
    ++g_backbone_stats.router_calls;
    return true;
}

bool gpu_hc_projection(const void* w, const float* x, float* out, int64_t rows, int64_t cols,
                        bool bf16, bool avx2_order) {
    if (!gpu_dense_projection(w, x, out, rows, cols, bf16, avx2_order)) return false;
    ++g_backbone_stats.hc_projection_calls;
    return true;
}

const void* gpu_resident(const void* host_data, size_t bytes) {
    return g_resident.get(host_data, bytes);
}

static bool g_gputime = getenv("DS41_GPU_TIMING") != nullptr;
static double g_t_res = 0, g_t_launch = 0, g_t_copy = 0;
static int g_tcalls = 0;
#define GT0() std::chrono::steady_clock::now()
#define GT_ACC(var, t0) var += std::chrono::duration<double>(std::chrono::steady_clock::now() - (t0)).count()
#define GT_MAYBE_PRINT() do { if (g_gputime && ++g_tcalls % 240 == 0) fprintf(stderr, "[gputime] calls=%d resident=%.3fs launch=%.3fs copy=%.3fs\n", g_tcalls, g_t_res, g_t_launch, g_t_copy); } while (0)

bool gpu_attn_prologue(const float* x, int64_t hidden,
                       const void* wq_a, const void* q_a_norm,
                       const void* wq_b,
                       const void* wkv, const void* kv_a_norm,
                       int64_t q_lora, int64_t head_dim, int64_t n_heads, int64_t rope_dim, float eps,
                       const float* cos_t, const float* sin_t,
                       float* qr_out, float* q_out, float* kv_out,
                       GpuAttentionState* resident, int layer, int64_t position) {
    if (resident && (!gpu_attention_ready(resident, layer, position) || resident->heads != n_heads ||
        resident->dim != head_dim || rope_dim < 0 || rope_dim > head_dim || rope_dim % 2)) return false;
    ExpertCache& c = cache();
    if (!c.ensure_input(hidden)) return false;
    cudaStream_t s = c.stream();

    auto gt = GT0();
    const void* d_wq_a = g_resident.get(wq_a, q8_matrix_bytes(q_lora, hidden));
    const void* d_wq_b = g_resident.get(wq_b, q8_matrix_bytes(n_heads * head_dim, q_lora));
    const void* d_wkv  = g_resident.get(wkv, q8_matrix_bytes(head_dim, hidden));
    const void* d_q_norm  = g_resident.get(q_a_norm, (size_t)q_lora * 4);
    const void* d_kv_norm = g_resident.get(kv_a_norm, (size_t)head_dim * 4);
    if (!d_wq_a || !d_wq_b || !d_wkv || !d_q_norm || !d_kv_norm) { if (g_gputime) fprintf(stderr, "[prologue] resident fail a=%p b=%p kv=%p qn=%p kvn=%p\n", d_wq_a, d_wq_b, d_wkv, d_q_norm, d_kv_norm); return false; }

    GT_ACC(g_t_res, gt);
    gt = GT0();
    float* d_qr = scratch_slot(g_attn.qr, g_attn.qr_n, q_lora);
    float* d_q  = resident ? resident->query : scratch_slot(g_attn.q, g_attn.q_n, n_heads * head_dim);
    float* d_kv = scratch_slot(g_attn.kv, g_attn.kv_n, head_dim);
    if (!d_qr || !d_q || !d_kv) { if (g_gputime) fprintf(stderr, "[prologue] scratch fail\n"); return false; }

    static const bool batch_readback = [] {
        const char* flag = getenv("DS41_QKV_READBACK_BATCH");
        return !flag || atoi(flag) != 0;
    }();
    const size_t qr_bytes = (size_t)q_lora * sizeof(float);
    const size_t q_bytes = resident ? 0 : (size_t)n_heads * head_dim * sizeof(float);
    const size_t kv_bytes = (size_t)head_dim * sizeof(float);
    const size_t readback_bytes = qr_bytes + q_bytes + kv_bytes;
    const size_t input_bytes = (size_t)hidden * sizeof(float);
    // Allocate before any upload: growing this shared pinned allocation while
    // H2D is in flight would invalidate its source. Stream order then permits
    // the same storage to receive all outputs without an intervening host wait.
    char* readback = pin_stage(std::max(readback_bytes, input_bytes));
    if (!readback) return false;
    auto fail = [&]() {
        cudaStreamSynchronize(s);
        return false;
    };

    // rope tables (static device buffers)
    const int half = (int)(rope_dim / 2);
    float* d_cos = resident ? resident->cos_t : rope_dev_cos();
    float* d_sin = resident ? resident->sin_t : rope_dev_sin();
    if (!d_cos || !d_sin) { if (g_gputime) fprintf(stderr, "[prologue] rope buffers fail\n"); return false; }
    if (cudaMemcpyAsync(d_cos, cos_t, half * 4, cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(d_sin, sin_t, half * 4, cudaMemcpyHostToDevice, s) != cudaSuccess) return fail();

    if (!h2d_stage(c.x_buf(), x, input_bytes, s)) return fail();

    const int rows_q = (int)(n_heads * head_dim);
    k_gemv_q8_0<<<(int)(q_lora * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint8_t*)d_wq_a, c.x_buf(), d_qr, (int)q_lora, (int)hidden);
    k_rms_norm_w<<<1, 256, 256 * 4, s>>>(d_qr, (const float*)d_q_norm, (int)q_lora, eps);
    k_gemv_q8_0<<<(rows_q * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint8_t*)d_wq_b, d_qr, d_q, rows_q, (int)q_lora);
    k_rope_norm_k<<<(int)(n_heads * half + 255) / 256, 256, 0, s>>>(
        d_q, (int)n_heads, (int)head_dim, (int)(head_dim - rope_dim), (int)rope_dim,
        (const float*)d_cos, (const float*)d_sin, 0);

    k_gemv_q8_0<<<(int)(head_dim * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint8_t*)d_wkv, c.x_buf(), d_kv, (int)head_dim, (int)hidden);
    k_rms_norm_w<<<1, 256, 256 * 4, s>>>(d_kv, (const float*)d_kv_norm, (int)head_dim, eps);
    k_rope_norm_k<<<(int)(1 * half + 255) / 256, 256, 0, s>>>(
        d_kv, 1, (int)head_dim, (int)(head_dim - rope_dim), (int)rope_dim,
        (const float*)d_cos, (const float*)d_sin, 0);

    GT_ACC(g_t_launch, gt);
    gt = GT0();
    if (cudaGetLastError() != cudaSuccess) return fail();
    if (resident && cudaMemcpyAsync(resident->raw + ((size_t)layer * resident->window + position % resident->window) * head_dim,
            d_kv, kv_bytes, cudaMemcpyDeviceToDevice, s) != cudaSuccess) return fail();
    if (batch_readback) {
        if (cudaMemcpyAsync(readback, d_qr, qr_bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            (q_bytes && cudaMemcpyAsync(readback + qr_bytes, d_q, q_bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) ||
            cudaMemcpyAsync(readback + qr_bytes + q_bytes, d_kv, kv_bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaStreamSynchronize(s) != cudaSuccess) return fail();
        memcpy(qr_out, readback, qr_bytes);
        if (q_bytes) memcpy(q_out, readback + qr_bytes, q_bytes);
        memcpy(kv_out, readback + qr_bytes + q_bytes, kv_bytes);
    } else if (!d2h_stage(qr_out, d_qr, qr_bytes, s) ||
               (q_bytes && !d2h_stage(q_out, d_q, q_bytes, s)) ||
               !d2h_stage(kv_out, d_kv, kv_bytes, s)) {
        return fail();
    }
    ++g_backbone_stats.prologue_calls;
    g_backbone_stats.prologue_readback_waits += batch_readback ? 1 : (resident ? 2 : 3);
    g_backbone_stats.prologue_readback_bytes += readback_bytes;
    if (resident) {
        resident->query_layer = layer; resident->query_position = position;
        resident->next_position[layer] = position + 1; resident->rope_dim = (int)rope_dim;
    }
    GT_ACC(g_t_copy, gt);
    GT_MAYBE_PRINT();
    return true;
}

static bool gpu_attn_tail_impl(const float* attn_out, bool device_input, int64_t head_dim, int64_t n_heads,
                   const void* wo_a, const void* wo_b,
                   int64_t o_groups, int64_t o_lora, int64_t hidden,
                   float* out) {
    if (!attn_out || !out || !wo_a || !wo_b || head_dim < 1 || n_heads < 1 || o_groups < 1 ||
        n_heads % o_groups || (n_heads / o_groups * head_dim) % 32 || o_lora < 1 || (o_groups * o_lora) % 32 || hidden < 1) return false;
    ExpertCache& c = cache();
    if (!c.ensure_device()) return false;
    cudaStream_t s = c.stream();

    const int64_t gdim = n_heads / o_groups * head_dim;
    const void* d_wo_a = g_resident.get(wo_a, q8_matrix_bytes(o_groups * o_lora, gdim));
    const void* d_wo_b = g_resident.get(wo_b, q8_matrix_bytes(hidden, o_groups * o_lora));
    if (!d_wo_a || !d_wo_b) return false;

    const float* d_in = device_input ? attn_out : scratch_slot(g_attn.q, g_attn.q_n, n_heads * head_dim);
    float* d_oa = scratch_slot(g_attn.oa, g_attn.oa_n, o_groups * o_lora);
    float* d_out = scratch_slot(g_attn.out, g_attn.out_n, hidden);
    if (!d_in || !d_oa || !d_out) return false;

    if (!device_input && !h2d_stage(const_cast<float*>(d_in), attn_out, (size_t)n_heads * head_dim * 4, s)) return false;

    static const bool grouped = [] {
        const char* value = getenv("DS41_GROUPED_TAIL");
        return !value || atoi(value) != 0;
    }();
    const int blocks = (int)(o_lora * 32 + kThreads - 1) / kThreads;
    const bool use_grouped = grouped && o_groups <= 65535;
    if (use_grouped) {
        // All group outputs must finish before wo_b, but there is no
        // dependency between groups. One grid replaces o_groups launches.
        k_gemv_q8_0<true><<<dim3(blocks, (unsigned)o_groups), kThreads, 0, s>>>(
            (const uint8_t*)d_wo_a, d_in, d_oa, (int)o_lora, (int)gdim);
    } else {
        const size_t a_row = (size_t)(gdim / 32) * 34u;
        for (int64_t gg = 0; gg < o_groups; gg++) {
            const uint8_t* wg = (const uint8_t*)d_wo_a + (size_t)gg * o_lora * a_row;
            const float* xg = d_in + (size_t)gg * gdim;
            float* yg = d_oa + (size_t)gg * o_lora;
            k_gemv_q8_0<<<blocks, kThreads, 0, s>>>(wg, xg, yg, (int)o_lora, (int)gdim);
        }
    }
    k_gemv_q8_0<<<(int)(hidden * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint8_t*)d_wo_b, d_oa, d_out, (int)hidden, (int)(o_groups * o_lora));
    if (!d2h_stage(out, d_out, (size_t)hidden * 4, s)) return false;
    ++g_backbone_stats.tail_calls;
    g_backbone_stats.grouped_tail_calls += use_grouped;
    g_backbone_stats.tail_gemv_launches += use_grouped ? 2 : (uint64_t)o_groups + 1;
    if (!device_input) g_backbone_stats.tail_upload_bytes += (size_t)n_heads * head_dim * 4;
    return true;
}

bool gpu_attn_tail(const float* attn_out, int64_t head_dim, int64_t n_heads,
                   const void* wo_a, const void* wo_b, int64_t o_groups,
                   int64_t o_lora, int64_t hidden, float* out) {
    return gpu_attn_tail_impl(attn_out, false, head_dim, n_heads, wo_a, wo_b, o_groups, o_lora, hidden, out);
}

bool gpu_attention_tail(GpuAttentionState* state, int layer, int64_t position,
                        const AttentionRows& rows, const float* sinks,
                        const void* wo_a, const void* wo_b, int64_t o_groups,
                        int64_t o_lora, int64_t hidden, float* out) {
    size_t comp_bytes = 0, metadata_bytes = 0;
    if (!enqueue_attention(state, layer, position, rows, sinks, comp_bytes, metadata_bytes)) return false;
    if (!gpu_attn_tail_impl(state->output, true, state->dim, state->heads, wo_a, wo_b, o_groups, o_lora, hidden, out)) {
        cudaStreamSynchronize(cache().stream());
        for (auto& c : state->compressed) c.uploaded = 0;
        return false;
    }
    ++g_backbone_stats.resident_attention_calls;
    g_backbone_stats.resident_comp_upload_bytes += comp_bytes;
    g_backbone_stats.resident_metadata_upload_bytes += metadata_bytes;
    g_backbone_stats.resident_attention_launches += 2;
    return true;
}

bool gpu_comp_state(const float* x, int64_t hidden, int64_t latent,
                    const void* comp_wkv, const void* comp_gate,
                    float* skv_out, float* sscore_out) {
    ExpertCache& c = cache();
    if (!c.ensure_input(hidden)) return false;
    cudaStream_t s = c.stream();
    const int64_t hd = latent;

    const void* d_wkv_g = g_resident.get(comp_wkv, (size_t)hidden * hd * 2);
    if (!d_wkv_g) return false;
    const void* d_gate = comp_gate ? g_resident.get(comp_gate, (size_t)hidden * hd * 2) : nullptr;
    if (comp_gate && !d_gate) return false;

    float* d_skv = scratch_slot(g_attn.qr, g_attn.qr_n, hd);
    float* d_sco = sscore_out ? scratch_slot(g_attn.kv, g_attn.kv_n, hd) : nullptr;
    if (!d_skv || (sscore_out && !d_sco)) return false;

    if (!h2d_stage(c.x_buf(), x, hidden * 4, s)) return false;
    k_gemv_bf16<<<(int)(hd * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint16_t*)d_wkv_g, c.x_buf(), d_skv, (int)hd, (int)hidden);
    if (d_gate) {
        k_gemv_bf16<<<(int)(hd * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            (const uint16_t*)d_gate, c.x_buf(), d_sco, (int)hd, (int)hidden);
    }
    if (!d2h_stage(skv_out, d_skv, hd * 4, s)) return false;
    if (sscore_out && !d2h_stage(sscore_out, d_sco, hd * 4, s)) return false;
    return true;
}

bool gpu_indexer_gemv(const void* w, const float* x, int64_t hidden_in, int64_t out_rows,
                      int dtypesel, float* out) {
    ExpertCache& c = cache();
    if (!c.ensure_input(hidden_in)) return false;
    cudaStream_t s = c.stream();

    float* d_out = scratch_slot(g_attn.oa, g_attn.oa_n, out_rows);
    if (!d_out) return false;
    if (!h2d_stage(c.x_buf(), x, hidden_in * 4, s)) return false;

    if (dtypesel == 0) {   // Q8_0
        size_t bytes = (size_t)out_rows * (size_t)(hidden_in / 32) * 34u;
        const void* d_w = g_resident.get(w, bytes);
        if (!d_w) return false;
        k_gemv_q8_0<<<(int)(out_rows * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            (const uint8_t*)d_w, c.x_buf(), d_out, (int)out_rows, (int)hidden_in);
    } else {               // BF16
        size_t bytes = (size_t)out_rows * (size_t)hidden_in * 2u;
        const void* d_w = g_resident.get(w, bytes);
        if (!d_w) return false;
        k_gemv_bf16<<<(int)(out_rows * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            (const uint16_t*)d_w, c.x_buf(), d_out, (int)out_rows, (int)hidden_in);
    }
    return d2h_stage(out, d_out, (size_t)out_rows * 4, s);
}

void gpu_print_stats() {
    ExpertCache& c = cache();
    if (!c.available()) return;
    unsigned long long tot = c.hits() + c.misses();
    fprintf(stderr, "ds41-gpu: expert cache: %llu lookups, %llu hits (%.1f%%), %llu evictions\n",
            tot, c.hits(), tot ? 100.0 * c.hits() / tot : 0.0, c.evictions());
}

} // namespace ds41
} // namespace nt
