// GLM-5.3-Flash GPU offload: resident backbone + LRU expert cache in VRAM.
// Reuses ds41's cache index (ds41_cache.h); Q8 GEMV kernels are batched
// variants of ds41_q8.cuh's q8_legacy with per-expert slot pointers.

#include "glm5_gpu.h"
#include "../ds41/ds41_cache.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
using HANDLE = void*;
#endif

namespace nt {
namespace glm5 {
namespace {

constexpr int kMaxUsed = 8;
constexpr int kThreads = 256;

struct Batch8 {
    const uint8_t* gate[kMaxUsed];
    const uint8_t* up[kMaxUsed];
    const uint8_t* down[kMaxUsed];
    float weights[kMaxUsed];
    int count;
};

// Batched Q8_0 GEMV with per-expert weight pointers.
// GateUp=true: gate@x -> yg, up@x -> yu (rows=ff, cols=hidden), one (k,tensor) per grid.y.
// GateUp=false: down@x -> yg (rows=hidden, cols=ff), one expert per grid.y.
template<bool GateUp>
__global__ void k_q8_batch(const Batch8 batch, const float* __restrict__ x,
                           float* __restrict__ yg, float* __restrict__ yu,
                           int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int k = GateUp ? (blockIdx.y >> 1) : blockIdx.y;
    const uint8_t* W = GateUp ? ((blockIdx.y & 1) ? batch.up[k] : batch.gate[k]) : batch.down[k];
    float* dst = GateUp ? (((blockIdx.y & 1) ? yu : yg) + (size_t)k * rows) : (yg + (size_t)k * rows);
    const float* xv = GateUp ? x : (x + (size_t)k * cols);   // down pass: per-expert h
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = xv + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) dst[wid] = acc;
}

__global__ void k_silu_clamp(const float* __restrict__ g, const float* __restrict__ u,
                             float* __restrict__ h, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = fminf(g[i], limit);
    const float uv = fmaxf(-limit, fminf(u[i], limit));
    h[i] = gv / (1.0f + __expf(-gv)) * uv;
}

__global__ void k_expert_sum(const Batch8 batch, const float* __restrict__ down,
                             float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
#pragma unroll 1
    for (int k = 0; k < batch.count; ++k) acc += batch.weights[k] * down[(size_t)k * n + i];
    out[i] = acc;
}

// Plain Q8_0 GEMV, warp per row: y[row] = W[row,:] @ x
__global__ void k_q8_single(const uint8_t* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
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

// Strided per-group Q8_0 GEMV: group g uses W + g*rows*(cols/32)*34,
// x + g*cols -> y + g*rows. For per-head absorbed MLA projections.
__global__ void k_q8_group_stride(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.y;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    const uint8_t* Wg = W + (size_t)g * rows * nb * 34u;
    const uint8_t* row = Wg + (size_t)wid * nb * 34u;
    const float* xg = x + (size_t)g * cols;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = xg + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)g * rows + wid] = acc;
}

// F32 matvec, warp per row
__global__ void k_f32_gemv(const float* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const float* row = W + (size_t)wid * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += row[c] * x[c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}

// Batched Q8_0 GEMV over T tokens: grid.y = token index.
// x is [T, cols], y is [T, rows], W is [rows, cols].
__global__ void k_q8_tokens(const uint8_t* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int t = blockIdx.y;
    const float* xt = x + (size_t)t * cols;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = xt + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)t * rows + wid] = acc;
}

// Batched F32 GEMV over T tokens.
__global__ void k_f32_tokens(const float* __restrict__ W, const float* __restrict__ x,
                             float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int t = blockIdx.y;
    const float* xt = x + (size_t)t * cols;
    const float* row = W + (size_t)wid * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += row[c] * xt[c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)t * rows + wid] = acc;
}

// Strided per-group Q8_0 GEMV over T tokens: grid = (rows, groups, T).
// x is [T, groups, cols], y is [T, groups, rows], W is [groups, rows, cols].
__global__ void k_q8_group_stride_t(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                    float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.y;
    const int t = blockIdx.z;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    const uint8_t* Wg = W + (size_t)g * rows * nb * 34u;
    const uint8_t* row = Wg + (size_t)wid * nb * 34u;
    const float* xg = x + ((size_t)t * gridDim.y + g) * cols;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t* blk = row + (size_t)b * 34u;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const float* xb = xg + (size_t)b * 32u;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
        acc += d * s;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[((size_t)t * gridDim.y + g) * rows + wid] = acc;
}

struct Glm5Gpu {
    cudaStream_t stream = nullptr;
    std::mutex mutex;

    std::unordered_map<const void*, void*> resident;
    size_t resident_bytes = 0;

    ds41::ExpertCacheIndex index;
    uint8_t* cache_base = nullptr;
    size_t slice_bytes = 0;
    size_t n_slots = 0;

    float* pin_x = nullptr;      size_t pin_x_cap = 0;
    float* pin_out = nullptr;    size_t pin_out_cap = 0;
    uint8_t** pin_ring = nullptr;
    static constexpr int ring_count = 24;
    int ring_next = 0;

    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_h = nullptr, *d_down = nullptr;
    int64_t d_hidden = 0, d_ff = 0, d_y_cap = 0;

    float* d_tx = nullptr;      // token-batched scratch
    float* d_ty = nullptr;
    size_t d_tx_cap = 0, d_ty_cap = 0;

    uint8_t* lm_w = nullptr;
    float *lm_x = nullptr, *lm_y = nullptr;
    int64_t lm_rows = 0, lm_cols = 0;
    float* lm_pin_x = nullptr;
    float* lm_pin_y = nullptr;

    uint64_t expert_calls = 0, expert_misses = 0;
    double seconds_stage = 0, seconds_kernels = 0;

    bool ensure_stream() {
        return stream || cudaStreamCreate(&stream) == cudaSuccess;
    }

    const void* dev(const void* host, size_t bytes) {
        auto it = resident.find(host);
        if (it != resident.end()) return it->second;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        if (cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            return nullptr;
        }
        resident.emplace(host, p);
        resident_bytes += bytes;
        return p;
    }

    float* stage_x(const float* x, size_t n) {
        if (pin_x_cap < n) {
            if (pin_x) cudaFreeHost(pin_x);
            pin_x_cap = 0;
            if (cudaMallocHost((void**)&pin_x, n * sizeof(float)) != cudaSuccess) return nullptr;
            pin_x_cap = n;
        }
        memcpy(pin_x, x, n * sizeof(float));
        return pin_x;
    }

    float* stage_out(size_t n) {
        if (pin_out_cap < n) {
            if (pin_out) cudaFreeHost(pin_out);
            pin_out_cap = 0;
            if (cudaMallocHost((void**)&pin_out, n * sizeof(float)) != cudaSuccess) return nullptr;
            pin_out_cap = n;
        }
        return pin_out;
    }

    bool ensure_tx(size_t n) {
        if (d_tx_cap >= n) return true;
        if (d_tx) cudaFree(d_tx);
        d_tx_cap = 0;
        if (cudaMalloc(&d_tx, n * 4) != cudaSuccess) return false;
        d_tx_cap = n;
        return true;
    }

    bool ensure_ty(size_t n) {
        if (d_ty_cap >= n) return true;
        if (d_ty) cudaFree(d_ty);
        d_ty_cap = 0;
        if (cudaMalloc(&d_ty, n * 4) != cudaSuccess) return false;
        d_ty_cap = n;
        return true;
    }

    bool ensure_y(int64_t n) {
        if (d_y_cap >= n) return true;
        if (d_y) cudaFree(d_y);
        d_y_cap = 0;
        if (cudaMalloc(&d_y, n * 4) != cudaSuccess) return false;
        d_y_cap = n;
        return true;
    }

    bool ensure_scratch(int64_t hidden, int64_t ff) {
        if (d_hidden >= hidden && d_ff >= ff) return true;
        if (d_x) {
            cudaFree(d_x); cudaFree(d_out);
            cudaFree(d_gate); cudaFree(d_up); cudaFree(d_h); cudaFree(d_down);
        }
        d_x = d_out = d_gate = d_up = d_h = d_down = nullptr;
        d_hidden = d_ff = 0;
        if (cudaMalloc(&d_x, hidden * 4) != cudaSuccess) return false;
        if (cudaMalloc(&d_out, hidden * 4) != cudaSuccess) return false;
        if (cudaMalloc(&d_gate, (size_t)kMaxUsed * ff * 4) != cudaSuccess) return false;
        if (cudaMalloc(&d_up, (size_t)kMaxUsed * ff * 4) != cudaSuccess) return false;
        if (cudaMalloc(&d_h, (size_t)kMaxUsed * ff * 4) != cudaSuccess) return false;
        if (cudaMalloc(&d_down, (size_t)kMaxUsed * hidden * 4) != cudaSuccess) return false;
        d_hidden = hidden;
        d_ff = ff;
        return true;
    }

    bool ensure_cache(size_t slice) {
        if (cache_base && slice_bytes == slice) return true;
        if (cache_base) return false;
        size_t budget;
        if (const char* env = getenv("GLM5_VRAM_GB")) {
            budget = (size_t)atoll(env) << 30;
        } else {
            size_t free_b = 0, total = 0;
            cudaMemGetInfo(&free_b, &total);
            // residents grow to ~6.6 GiB during the run and WDDM needs ~1 GiB
            // of headroom beyond cache+residents: a larger budget silently
            // thrashes (device paging to system memory)
            budget = free_b > (85ull << 30) / 10 ? free_b - (85ull << 30) / 10 : 0;
        }
        n_slots = budget / slice;
        if (n_slots < 64) {
            fprintf(stderr, "glm5-gpu: expert cache budget too small (%zu slots)\n", n_slots);
            return false;
        }
        if (cudaMalloc(&cache_base, n_slots * slice) != cudaSuccess) {
            cache_base = nullptr;
            return false;
        }
        slice_bytes = slice;
        index.configure(n_slots);
        pin_ring = new uint8_t*[ring_count];
        for (int i = 0; i < ring_count; i++) pin_ring[i] = nullptr;
        fprintf(stderr, "glm5-gpu: expert cache %.2f GiB = %zu slots of %.1f MiB\n",
                (double)(n_slots * slice) / (1ull << 30), n_slots, slice / 1048576.0);
        return true;
    }

    uint8_t* slot_ptr(size_t slot) { return cache_base + slot * slice_bytes; }

    static uint64_t make_key(const void* base, int expert) {
        return (uint64_t)(uintptr_t)base * 1000003ull + (uint64_t)expert * 2654435761ull;
    }

    ~Glm5Gpu() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto& kv : resident) cudaFree(kv.second);
        if (cache_base) cudaFree(cache_base);
        if (d_x) {
            cudaFree(d_x); cudaFree(d_out);
            cudaFree(d_gate); cudaFree(d_up); cudaFree(d_h); cudaFree(d_down);
        }
        if (d_y) cudaFree(d_y);
        if (d_tx) cudaFree(d_tx);
        if (d_ty) cudaFree(d_ty);
        if (lm_w) { cudaFree(lm_w); cudaFree(lm_x); cudaFree(lm_y); }
        if (lm_pin_x) cudaFreeHost(lm_pin_x);
        if (lm_pin_y) cudaFreeHost(lm_pin_y);
        if (pin_x) cudaFreeHost(pin_x);
        if (pin_out) cudaFreeHost(pin_out);
        if (pin_ring) {
            for (int i = 0; i < ring_count; i++) if (pin_ring[i]) cudaFreeHost(pin_ring[i]);
            delete[] pin_ring;
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

Glm5Gpu& g() {
    static Glm5Gpu inst;
    return inst;
}

// ------------------------------------------------------------------
// pipelined expert engine: disk fills land in pinned RAM rings via a
// persistent thread pool; expert calls H2D from the ring. The layer-ahead
// prefetch submits fills for the next layer while the current one runs.
// ------------------------------------------------------------------
struct ExpertEngine {
    struct Task { int ring; const uint8_t* src; bool urgent; };
    std::mutex m;
    std::condition_variable cv_work, cv_ready;
    std::deque<Task> tasks;
    std::vector<std::thread> workers;
    bool stop = false;

    // shard registry: host range -> file to read the bytes from
    struct ShardRange {
        const uint8_t* base;
        size_t length;
        uint64_t file_data_offset;
        std::string path;
    };
    std::vector<ShardRange> shards;

#ifdef _WIN32
    // per-worker open handles, [worker][shard]
    std::vector<std::vector<HANDLE>> worker_handles;
    std::atomic<uint64_t> readfile_bytes{0};
    std::atomic<uint64_t> readfile_calls{0};
#endif
    // fill classification: fast (< 1.5ms) = page-cache hit, slow = disk
    std::atomic<uint64_t> fills_fast{0}, fills_slow{0};
    std::atomic<uint64_t> fast_ns{0}, slow_ns{0};

    struct Ring {
        uint8_t* buf = nullptr;
        std::atomic<int> state{0};   // 0 free, 1 filling, 2 ready, 3 h2d
        uint64_t key = 0;
        int last_use = 0;
    };
    std::unique_ptr<Ring[]> rings;
    int n_rings = 0;
    std::unordered_map<uint64_t, int> by_key;
    size_t slice = 0;
    int use_counter = 0;
    uint64_t pf_submitted = 0, pf_used = 0, pf_stolen = 0;
    double fill_seconds = 0;

    void ensure(size_t slice_bytes) {
        if (rings) return;
        int n = 96;
        if (const char* e = getenv("GLM5_RINGS")) n = atoi(e);
        if (n < 24) n = 24;
        if (n > 256) n = 256;
        rings.reset(new Ring[n]);
        n_rings = n;
        for (int i = 0; i < n; i++) {
            if (cudaMallocHost((void**)&rings[i].buf, slice_bytes) != cudaSuccess) rings[i].buf = nullptr;
        }
        slice = slice_bytes;
        int nw = 8;
        if (const char* e = getenv("GLM5_FILL_THREADS")) nw = atoi(e);
        if (nw < 1) nw = 1;
        if (nw > 16) nw = 16;
        for (int i = 0; i < nw; i++) workers.emplace_back([this, i] { worker(i); });
    }

    void worker(int wid) {
        for (;;) {
            Task t;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_work.wait(lk, [&] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                t = tasks.front();
                tasks.pop_front();
            }
            const auto t0 = std::chrono::steady_clock::now();
            fill_slice(wid, t.ring, t.src);
            const auto t1 = std::chrono::steady_clock::now();
            const int64_t ns = (int64_t)std::chrono::duration<double, std::nano>(t1 - t0).count();
            {
                std::lock_guard<std::mutex> lk(m);
                rings[t.ring].state = 2;
                fill_seconds += std::chrono::duration<double>(t1 - t0).count();
            }
            if (ns < 1500000) { ++fills_fast; fast_ns += ns; }
            else { ++fills_slow; slow_ns += ns; }
            cv_ready.notify_all();
        }
    }

    // read one expert slice into the ring: ReadFile when the shard is
    // registered, page-fault memcpy otherwise
    void fill_slice(int wid, int ring, const uint8_t* src) {
        uint8_t* dst = rings[ring].buf;
#ifdef _WIN32
        {
            std::lock_guard<std::mutex> lk(m);
            if (worker_handles.size() <= (size_t)wid) worker_handles.resize(wid + 1);
            auto& handles = worker_handles[wid];
            for (size_t si = 0; si < shards.size(); si++) {
                const ShardRange& sr = shards[si];
                if (src < sr.base || src >= sr.base + sr.length) continue;
                if (handles.size() <= si) handles.resize(si + 1, INVALID_HANDLE_VALUE);
                if (handles[si] == INVALID_HANDLE_VALUE) {
                    handles[si] = CreateFileA(sr.path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                }
                HANDLE h = handles[si];
                if (h != INVALID_HANDLE_VALUE) {
                    const uint64_t file_off = sr.file_data_offset + (uint64_t)(src - sr.base);
                    uint8_t* p = dst;
                    uint64_t remaining = slice;
                    uint8_t* base_pos = (uint8_t*)src;
                    OVERLAPPED ov = {};
                    while (remaining > 0) {
                        const uint64_t off = sr.file_data_offset + (uint64_t)(base_pos - sr.base) + (slice - remaining);
                        ov.Offset = (DWORD)(off & 0xFFFFFFFFull);
                        ov.OffsetHigh = (DWORD)(off >> 32);
                        const DWORD chunk = (DWORD)std::min<uint64_t>(remaining, 1 << 22);
                        DWORD got = 0;
                        if (!ReadFile(h, p, chunk, &got, &ov) || got != chunk) {
                            // fall back to memcpy for the remainder
                            memcpy(p, src + (slice - remaining), remaining);
                            break;
                        }
                        p += got;
                        remaining -= got;
                    }
                    readfile_bytes += slice - remaining;
                    ++readfile_calls;
                    return;
                }
            }
        }
#endif
        memcpy(dst, src, slice);
    }

    // submit a fill without waiting; returns ring index
    int submit(uint64_t key, const uint8_t* src) {
        std::unique_lock<std::mutex> lk(m);
        auto it = by_key.find(key);
        if (it != by_key.end()) return it->second;
        int r = pick_ring_locked(lk);
        rings[r].key = key;
        rings[r].state = 1;
        rings[r].last_use = ++use_counter;
        by_key[key] = r;
        tasks.push_front({r, src, true});   // demand fill: highest priority
        cv_work.notify_one();
        return r;
    }

    void wait_ready(int r) {
        std::unique_lock<std::mutex> lk(m);
        cv_ready.wait(lk, [&] { return rings[r].state >= 2; });
        rings[r].last_use = ++use_counter;
    }

    // returns ring index with state >= 2 (ready). call without holding m.
    int acquire(uint64_t key, const uint8_t* src) {
        int r = submit(key, src);
        wait_ready(r);
        return r;
    }

    // non-blocking prefetch submit; drops when no ring is free/stealable
    void submit_prefetch(uint64_t key, const uint8_t* src) {
        std::lock_guard<std::mutex> lk(m);
        if (by_key.count(key)) return;
        if (tasks.size() >= 96) return;
        int r = -1;
        int free_cnt = 0;
        for (int i = 0; i < n_rings; i++) {
            if (rings[i].state == 0) {
                ++free_cnt;
                if (r < 0) r = i;
            }
        }
        if (free_cnt < 24) r = -1;   // keep a reserve for demand fills
        if (r < 0) {
            int best = -1, best_use = INT_MAX;
            for (int i = 0; i < n_rings; i++) {
                if (rings[i].state == 2 && rings[i].last_use < best_use) { best_use = rings[i].last_use; best = i; }
            }
            if (best < 0) return;
            r = best;
            by_key.erase(rings[r].key);
            ++pf_stolen;
        }
        rings[r].key = key;
        rings[r].state = 1;
        rings[r].last_use = ++use_counter;
        by_key[key] = r;
        tasks.push_back({r, src, false});
        ++pf_submitted;
        cv_work.notify_one();
    }

    int pick_ring_locked(std::unique_lock<std::mutex>& lk) {
        for (;;) {
            for (int i = 0; i < n_rings; i++) {
                if (rings[i].state == 0) return i;
            }
            int best = -1, best_use = INT_MAX;
            for (int i = 0; i < n_rings; i++) {
                if (rings[i].state == 2 && rings[i].last_use < best_use) { best_use = rings[i].last_use; best = i; }
            }
            if (best >= 0) {
                by_key.erase(rings[best].key);
                return best;
            }
            // everything is filling or queued for h2d: wait for progress
            cv_ready.wait(lk);
        }
    }

    // reclaim rings whose H2D was issued in a previous (already synced) call
    void reclaim_h2d() {
        std::lock_guard<std::mutex> lk(m);
        for (int i = 0; i < n_rings; i++) {
            if (rings[i].state == 3) {
                by_key.erase(rings[i].key);
                rings[i].state = 0;
                rings[i].key = 0;
            }
        }
    }

    ~ExpertEngine() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv_work.notify_all();
        for (auto& w : workers) w.join();
#ifdef _WIN32
        for (auto& hs : worker_handles)
            for (HANDLE h : hs)
                if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#endif
        for (int i = 0; i < n_rings; i++) if (rings[i].buf) cudaFreeHost(rings[i].buf);
    }
};

ExpertEngine& eng() {
    static ExpertEngine inst;
    return inst;
}

} // namespace

void glm5_register_shard(const void* base, size_t length, const char* path, uint64_t file_data_offset) {
    ExpertEngine& e = eng();
    std::lock_guard<std::mutex> lk(e.m);
    for (auto& s : e.shards) {
        if (s.base == base) return;
    }
    ExpertEngine::ShardRange sr;
    sr.base = (const uint8_t*)base;
    sr.length = length;
    sr.file_data_offset = file_data_offset;
    sr.path = path;
    e.shards.push_back(sr);
}

bool glm5_gpu_enabled() {
    static const bool on = [] {
        const char* e = getenv("GLM5_GPU");
        return e ? atoi(e) != 0 : false;
    }();
    if (!on) return false;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return false;
    return g().ensure_stream();
}

const void* glm5_resident(const void* host, size_t bytes) {
    if (!glm5_gpu_enabled()) return nullptr;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.dev(host, bytes);
}

bool glm5_q8_proj(const void* w, const float* x, float* y, int64_t rows, int64_t cols) {
    if (!glm5_gpu_enabled()) return false;
    if (rows <= 0 || cols <= 0 || cols % 32 || rows > 65536 || cols > 65536) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const uint8_t* dw = (const uint8_t*)s.dev(w, (size_t)rows * (cols / 32) * 34);
    float* px = s.stage_x(x, cols);
    if (!dw || !px || !s.ensure_scratch(cols, 4) || !s.ensure_y(rows)) return false;
    cudaMemcpyAsync(s.d_x, px, cols * 4, cudaMemcpyHostToDevice, s.stream);
    k_q8_single<<<(int)((rows * 32 + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
        dw, s.d_x, s.d_y, (int)rows, (int)cols);
    float* po = s.stage_out(rows);
    if (!po) return false;
    cudaMemcpyAsync(po, s.d_y, rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, rows * 4);
    return true;
}

bool glm5_q8_proj_grouped(const void* w, const float* x, float* y,
                          int64_t rows, int64_t cols, int groups) {
    if (!glm5_gpu_enabled()) return false;
    if (rows <= 0 || cols <= 0 || cols % 32 || rows > 65536 || cols > 65536) return false;
    if (groups <= 0 || groups > 64) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const uint8_t* dw = (const uint8_t*)s.dev(w, (size_t)groups * rows * (cols / 32) * 34);
    float* px = s.stage_x(x, (size_t)groups * cols);
    if (!dw || !px || !s.ensure_scratch(groups * cols, 4) || !s.ensure_y((size_t)groups * rows)) return false;
    float* po = s.stage_out((size_t)groups * rows);
    if (!po) return false;
    cudaMemcpyAsync(s.d_x, px, (size_t)groups * cols * 4, cudaMemcpyHostToDevice, s.stream);
    dim3 grid((unsigned)((rows * 32 + kThreads - 1) / kThreads), (unsigned)groups);
    k_q8_group_stride<<<grid, kThreads, 0, s.stream>>>(dw, s.d_x, s.d_y, (int)rows, (int)cols);
    cudaMemcpyAsync(po, s.d_y, (size_t)groups * rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, (size_t)groups * rows * 4);
    return true;
}

bool glm5_f32_proj(const void* w, const float* x, float* y, int64_t rows, int64_t cols) {
    if (!glm5_gpu_enabled()) return false;
    if (rows <= 0 || cols <= 0 || rows > 65536 || cols > 65536) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const float* dw = (const float*)s.dev(w, (size_t)rows * cols * 4);
    float* px = s.stage_x(x, cols);
    if (!dw || !px || !s.ensure_scratch(cols, 4)) return false;
    cudaMemcpyAsync(s.d_x, px, cols * 4, cudaMemcpyHostToDevice, s.stream);
    k_f32_gemv<<<(int)((rows * 32 + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
        dw, s.d_x, s.d_out, (int)rows, (int)cols);
    float* po = s.stage_out(rows);
    if (!po) return false;
    cudaMemcpyAsync(po, s.d_out, rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, rows * 4);
    return true;
}

bool glm5_q8_proj_many(const void* const* mats, const int64_t* rows, int n_mats,
                       int64_t cols, const float* x, float* y) {
    if (!glm5_gpu_enabled() || n_mats <= 0 || n_mats > 8) return false;
    if (cols <= 0 || cols % 32 || cols > 65536) return false;
    int64_t total_rows = 0;
    for (int i = 0; i < n_mats; i++) {
        if (rows[i] <= 0 || rows[i] > 65536) return false;
        total_rows += rows[i];
    }
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    float* px = s.stage_x(x, cols);
    if (!px || !s.ensure_scratch(cols, 4) || !s.ensure_y(total_rows)) return false;
    float* po = s.stage_out(total_rows);
    if (!po) return false;
    cudaMemcpyAsync(s.d_x, px, cols * 4, cudaMemcpyHostToDevice, s.stream);
    int64_t row_off = 0;
    for (int i = 0; i < n_mats; i++) {
        const uint8_t* dw = (const uint8_t*)s.dev(mats[i], (size_t)rows[i] * (cols / 32) * 34);
        if (!dw) return false;
        k_q8_single<<<(int)((rows[i] * 32 + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
            dw, s.d_x, s.d_y + row_off, (int)rows[i], (int)cols);
        row_off += rows[i];
    }
    cudaMemcpyAsync(po, s.d_y, total_rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, total_rows * 4);
    return true;
}

bool glm5_q8_proj_tokens(const void* w, const float* x, float* y,
                         int64_t rows, int64_t cols, int T) {
    if (!glm5_gpu_enabled() || T <= 0 || T > 1024) return false;
    if (rows <= 0 || cols <= 0 || cols % 32 || rows > 65536 || cols > 65536) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const uint8_t* dw = (const uint8_t*)s.dev(w, (size_t)rows * (cols / 32) * 34);
    if (!dw) return false;
    if (!s.ensure_tx((size_t)T * cols) || !s.ensure_ty((size_t)T * rows)) return false;
    float* px = s.stage_x(x, (size_t)T * cols);
    float* po = s.stage_out((size_t)T * rows);
    if (!px || !po) return false;
    cudaMemcpyAsync(s.d_tx, px, (size_t)T * cols * 4, cudaMemcpyHostToDevice, s.stream);
    dim3 grid((unsigned)((rows * 32 + kThreads - 1) / kThreads), (unsigned)T);
    k_q8_tokens<<<grid, kThreads, 0, s.stream>>>(dw, s.d_tx, s.d_ty, (int)rows, (int)cols);
    cudaMemcpyAsync(po, s.d_ty, (size_t)T * rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, (size_t)T * rows * 4);
    return true;
}

bool glm5_f32_proj_tokens(const void* w, const float* x, float* y,
                          int64_t rows, int64_t cols, int T) {
    if (!glm5_gpu_enabled() || T <= 0 || T > 1024) return false;
    if (rows <= 0 || cols <= 0 || rows > 65536 || cols > 65536) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const float* dw = (const float*)s.dev(w, (size_t)rows * cols * 4);
    if (!dw) return false;
    if (!s.ensure_tx((size_t)T * cols) || !s.ensure_ty((size_t)T * rows)) return false;
    float* px = s.stage_x(x, (size_t)T * cols);
    float* po = s.stage_out((size_t)T * rows);
    if (!px || !po) return false;
    cudaMemcpyAsync(s.d_tx, px, (size_t)T * cols * 4, cudaMemcpyHostToDevice, s.stream);
    dim3 grid((unsigned)((rows * 32 + kThreads - 1) / kThreads), (unsigned)T);
    k_f32_tokens<<<grid, kThreads, 0, s.stream>>>(dw, s.d_tx, s.d_ty, (int)rows, (int)cols);
    cudaMemcpyAsync(po, s.d_ty, (size_t)T * rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, (size_t)T * rows * 4);
    return true;
}

bool glm5_q8_proj_grouped_t(const void* w, const float* x, float* y,
                            int64_t rows, int64_t cols, int groups, int T) {
    if (!glm5_gpu_enabled() || T <= 0 || T > 1024 || groups <= 0 || groups > 64) return false;
    if (rows <= 0 || cols <= 0 || cols % 32 || rows > 65536 || cols > 65536) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const uint8_t* dw = (const uint8_t*)s.dev(w, (size_t)groups * rows * (cols / 32) * 34);
    if (!dw) return false;
    const size_t nx = (size_t)T * groups * cols;
    const size_t ny = (size_t)T * groups * rows;
    if (!s.ensure_tx(nx) || !s.ensure_ty(ny)) return false;
    float* px = s.stage_x(x, nx);
    float* po = s.stage_out(ny);
    if (!px || !po) return false;
    cudaMemcpyAsync(s.d_tx, px, nx * 4, cudaMemcpyHostToDevice, s.stream);
    dim3 grid((unsigned)((rows * 32 + kThreads - 1) / kThreads), (unsigned)groups, (unsigned)T);
    k_q8_group_stride_t<<<grid, kThreads, 0, s.stream>>>(dw, s.d_tx, s.d_ty, (int)rows, (int)cols);
    cudaMemcpyAsync(po, s.d_ty, ny * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(y, po, ny * 4);
    return true;
}

bool glm5_gpu_experts(const float* x, int64_t hidden,
                      const void* gate_base, const void* up_base, const void* down_base,
                      int64_t ff, const int* experts, const float* weights, int n_used,
                      float limit, float* out,
                      const std::function<void()>& host_work) {
    if (!glm5_gpu_enabled()) return false;
    if (hidden <= 0 || hidden > 8192 || hidden % 32 || ff <= 0 || ff > 8192 || ff % 32) return false;
    if (n_used <= 0 || n_used > kMaxUsed) return false;

    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);

    const size_t gate_slice = (size_t)ff * (hidden / 32) * 34;
    const size_t down_slice = (size_t)hidden * (ff / 32) * 34;
    const size_t slice = std::max(gate_slice, down_slice);
    if (!s.ensure_scratch(hidden, ff)) return false;
    if (!s.ensure_cache(slice)) return false;

    ExpertEngine& e = eng();
    e.ensure(slice);
    e.reclaim_h2d();   // prior call ended with a stream sync

    const auto t0 = std::chrono::steady_clock::now();
    ++s.expert_calls;

    float* px = s.stage_x(x, hidden);
    if (!px) return false;
    cudaMemcpyAsync(s.d_x, px, hidden * 4, cudaMemcpyHostToDevice, s.stream);

    s.index.begin_batch();
    Batch8 batch;
    batch.count = n_used;

    const uint8_t* bases[3] = { (const uint8_t*)gate_base, (const uint8_t*)up_base, (const uint8_t*)down_base };
    uint8_t* slot_of[3][kMaxUsed];

    {
        // pass 1: LRU lookups (reserve slots) and parallel fill submission
        struct RingUse { int ring; uint8_t* slot; };
        std::vector<RingUse> ring_uses;
        for (int k = 0; k < n_used; k++) {
            for (int t = 0; t < 3; t++) {
                const uint64_t key = s.make_key(bases[t], experts[k]);
                auto r = s.index.lookup_or_reserve(key);
                slot_of[t][k] = s.slot_ptr(r.slot);
                if (!r.hit) {
                    ++s.expert_misses;
                    ring_uses.push_back({-1, slot_of[t][k]});
                    ring_uses.back().ring = e.submit(key, bases[t] + (size_t)experts[k] * slice);
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        // wait for all fills (they ran in parallel), then H2D
        for (auto& ru : ring_uses) e.wait_ready(ru.ring);
        for (auto& ru : ring_uses) {
            cudaMemcpyAsync(ru.slot, e.rings[ru.ring].buf, slice, cudaMemcpyHostToDevice, s.stream);
            e.rings[ru.ring].state = 3;
        }
        s.seconds_stage += std::chrono::duration<double>(t1 - t0).count();
    }

    for (int k = 0; k < n_used; k++) {
        batch.gate[k] = slot_of[0][k];
        batch.up[k] = slot_of[1][k];
        batch.down[k] = slot_of[2][k];
        batch.weights[k] = weights[k];
    }

    {
        dim3 grid((unsigned)((ff * 32 + kThreads - 1) / kThreads), (unsigned)(2 * n_used));
        k_q8_batch<true><<<grid, kThreads, 0, s.stream>>>(
            batch, s.d_x, s.d_gate, s.d_up, (int)ff, (int)hidden);
    }
    k_silu_clamp<<<(int)((ff * n_used + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
        s.d_gate, s.d_up, s.d_h, (int)(ff * n_used), limit);
    {
        dim3 grid((unsigned)((hidden * 32 + kThreads - 1) / kThreads), (unsigned)n_used);
        k_q8_batch<false><<<grid, kThreads, 0, s.stream>>>(
            batch, s.d_h, s.d_down, nullptr, (int)hidden, (int)ff);
    }
    k_expert_sum<<<(int)((hidden + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
        batch, s.d_down, s.d_out, (int)hidden);

    float* po = s.stage_out(hidden);
    if (!po) return false;
    cudaMemcpyAsync(po, s.d_out, hidden * 4, cudaMemcpyDeviceToHost, s.stream);

    if (host_work) host_work();

    cudaStreamSynchronize(s.stream);
    memcpy(out, po, hidden * 4);

    s.seconds_kernels += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                       - s.seconds_stage;   // approximation; fine for stats
    return true;
}

bool glm5_lm_head_init(const void* w, int64_t rows, int64_t cols) {
    if (!glm5_gpu_enabled()) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (s.lm_w) return true;
    const size_t bytes = (size_t)rows * (cols / 32) * 34;
    if (cudaMalloc(&s.lm_w, bytes) != cudaSuccess) return false;
    if (cudaMemcpy(s.lm_w, w, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMalloc(&s.lm_x, cols * 4) || cudaMalloc(&s.lm_y, rows * 4)) return false;
    if (cudaMallocHost((void**)&s.lm_pin_x, cols * 4) != cudaSuccess) return false;
    if (cudaMallocHost((void**)&s.lm_pin_y, rows * 4) != cudaSuccess) return false;
    s.lm_rows = rows;
    s.lm_cols = cols;
    return true;
}

bool glm5_lm_head(const float* x, float* logits, int64_t rows, int64_t cols) {
    if (!glm5_gpu_enabled()) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.lm_w || s.lm_rows != rows || s.lm_cols != cols) return false;
    memcpy(s.lm_pin_x, x, cols * 4);
    cudaMemcpyAsync(s.lm_x, s.lm_pin_x, cols * 4, cudaMemcpyHostToDevice, s.stream);

    Batch8 batch;
    memset(&batch, 0, sizeof(batch));
    batch.count = 1;
    batch.gate[0] = s.lm_w;
    batch.weights[0] = 1.0f;
    k_q8_single<<<(int)((rows * 32 + kThreads - 1) / kThreads), kThreads, 0, s.stream>>>(
        s.lm_w, s.lm_x, s.lm_y, (int)rows, (int)cols);

    cudaMemcpyAsync(s.lm_pin_y, s.lm_y, rows * 4, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    memcpy(logits, s.lm_pin_y, rows * 4);
    return true;
}

void glm5_gpu_stats() {
    Glm5Gpu& s = g();
    const auto& st = s.index.stats();
    ExpertEngine& e = eng();
    fprintf(stderr, "[glm5-gpu] residents=%.2f GiB cache=%.2f GiB (%zu/%zu slots) "
            "calls=%llu misses=%llu hits=%llu evict=%llu stage=%.2fs kernels=%.2fs\n",
            (double)s.resident_bytes / (1ull << 30),
            (double)(s.n_slots * s.slice_bytes) / (1ull << 30),
            s.index.size(), s.n_slots,
            (unsigned long long)s.expert_calls, (unsigned long long)s.expert_misses,
            (unsigned long long)st.hits, (unsigned long long)st.evictions,
            s.seconds_stage, s.seconds_kernels);
    fprintf(stderr, "[glm5-gpu] prefetch: submitted=%llu stolen=%llu fill=%.1fs (%.2f GiB/s)\n",
            (unsigned long long)e.pf_submitted, (unsigned long long)e.pf_stolen,
            e.fill_seconds,
            e.fill_seconds > 0 ? e.pf_submitted * eng().slice / e.fill_seconds / (1ull << 30) : 0.0);
    const double fast_s = (double)e.fast_ns / 1e9, slow_s = (double)e.slow_ns / 1e9;
    const double fast_g = (double)e.fills_fast * e.slice / (1ull << 30);
    const double slow_g = (double)e.fills_slow * e.slice / (1ull << 30);
    const uint64_t tot = e.fills_fast + e.fills_slow;
    fprintf(stderr, "[glm5-gpu] fills: cache-hit=%llu (%.1f GiB @ %.2f GiB/s) disk=%llu (%.1f GiB @ %.2f GiB/s) hit-rate=%.1f%%\n",
            (unsigned long long)e.fills_fast, fast_g, fast_s > 0 ? fast_g / fast_s : 0.0,
            (unsigned long long)e.fills_slow, slow_g, slow_s > 0 ? slow_g / slow_s : 0.0,
            tot ? 100.0 * (double)e.fills_fast / (double)tot : 0.0);
}

void glm5_gpu_load_experts(const void* gate_base, const void* up_base, const void* down_base,
                           int64_t ff, const int* experts, int n) {
    if (!glm5_gpu_enabled() || n <= 0) return;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const size_t gate_slice = (size_t)ff * (4096 / 32) * 34;
    const size_t down_slice = (size_t)4096 * (ff / 32) * 34;
    const size_t slice = std::max(gate_slice, down_slice);
    if (!s.ensure_cache(slice)) return;
    ExpertEngine& e = eng();
    e.ensure(slice);
    s.index.begin_batch();
    const uint8_t* bases[3] = { (const uint8_t*)gate_base, (const uint8_t*)up_base, (const uint8_t*)down_base };
    struct RU { int ring; uint8_t* slot; };
    std::vector<RU> rus;
    for (int k = 0; k < n; k++) {
        for (int t = 0; t < 3; t++) {
            const uint64_t key = s.make_key(bases[t], experts[k]);
            auto r = s.index.lookup_or_reserve(key);
            if (!r.hit) {
                ++s.expert_misses;
                rus.push_back({e.submit(key, bases[t] + (size_t)experts[k] * slice), s.slot_ptr(r.slot)});
            }
        }
    }
    for (auto& ru : rus) e.wait_ready(ru.ring);
    for (auto& ru : rus) {
        cudaMemcpyAsync(ru.slot, e.rings[ru.ring].buf, slice, cudaMemcpyHostToDevice, s.stream);
        e.rings[ru.ring].state = 3;
    }
    cudaStreamSynchronize(s.stream);
    e.reclaim_h2d();
}

bool glm5_gpu_stage_experts(const void* gate_base, const void* up_base, const void* down_base,
                            int64_t ff, const int* experts, int n,
                            void* d_etable, void* user_stream) {
    if (!glm5_gpu_enabled() || n <= 0 || n > kMaxUsed) return false;
    Glm5Gpu& s = g();
    std::lock_guard<std::mutex> lk(s.mutex);
    const size_t gate_slice = (size_t)ff * (4096 / 32) * 34;
    const size_t down_slice = (size_t)4096 * (ff / 32) * 34;
    const size_t slice = std::max(gate_slice, down_slice);
    if (!s.ensure_cache(slice)) return false;
    ExpertEngine& e = eng();
    e.ensure(slice);
    e.reclaim_h2d();   // los H2D del stage anterior ya completaron (hubo sync del router)
    s.index.begin_batch();
    const uint8_t* bases[3] = { (const uint8_t*)gate_base, (const uint8_t*)up_base, (const uint8_t*)down_base };
    struct RU { int ring; uint8_t* slot; };
    std::vector<RU> rus;
    for (int k = 0; k < n; k++) {
        for (int t = 0; t < 3; t++) {
            const uint64_t key = s.make_key(bases[t], experts[k]);
            auto r = s.index.lookup_or_reserve(key);
            if (!r.hit) {
                ++s.expert_misses;
                rus.push_back({e.submit(key, bases[t] + (size_t)experts[k] * slice), s.slot_ptr(r.slot)});
            } else {
                rus.push_back({-1, s.slot_ptr(r.slot)});
            }
        }
    }
    for (auto& ru : rus) {
        if (ru.ring >= 0) e.wait_ready(ru.ring);
    }
    cudaStream_t us = (cudaStream_t)user_stream;
    for (auto& ru : rus) {
        if (ru.ring >= 0) {
            cudaMemcpyAsync(ru.slot, e.rings[ru.ring].buf, slice, cudaMemcpyHostToDevice, us);
            e.rings[ru.ring].state = 3;
        }
    }
    // tabla de punteros [gate[8], up[8], down[8]]
    static uint64_t* pin_tab = [] {
        uint64_t* p = nullptr;
        cudaMallocHost((void**)&p, 24 * sizeof(uint64_t));
        return p;
    }();
    if (!pin_tab) return false;
    for (int k = 0; k < n; k++) {
        pin_tab[k] = (uint64_t)(uintptr_t)rus[(size_t)k * 3 + 0].slot;              // gate
        pin_tab[n + k] = (uint64_t)(uintptr_t)rus[(size_t)k * 3 + 1].slot;           // up
        pin_tab[2 * n + k] = (uint64_t)(uintptr_t)rus[(size_t)k * 3 + 2].slot;       // down
    }
    for (int k = n; k < kMaxUsed; k++) {
        pin_tab[k] = pin_tab[0];
        pin_tab[kMaxUsed + k] = pin_tab[kMaxUsed];
        pin_tab[2 * kMaxUsed + k] = pin_tab[2 * kMaxUsed];
    }
    cudaMemcpyAsync(d_etable, pin_tab, 24 * sizeof(uint64_t), cudaMemcpyHostToDevice, us);
    // los rings quedan en estado 3; el engine los reclama en el proximo fill
    return true;
}

void glm5_prefetch_layer(const void* gate_base, const void* up_base, const void* down_base,
                         int64_t slice, const int* experts, int n) {
    if (!glm5_gpu_enabled()) return;
    ExpertEngine& e = eng();
    if (e.slice != (size_t)slice || !e.rings) return;
    const uint8_t* bases[3] = { (const uint8_t*)gate_base, (const uint8_t*)up_base, (const uint8_t*)down_base };
    for (int k = 0; k < n; k++) {
        for (int t = 0; t < 3; t++) {
            const uint64_t key = (uint64_t)(uintptr_t)bases[t] * 1000003ull + (uint64_t)experts[k] * 2654435761ull;
            // skip slices already resident in the VRAM cache
            {
                Glm5Gpu& s = g();
                std::lock_guard<std::mutex> lk(s.mutex);
                if (s.index.contains(key)) continue;
            }
            e.submit_prefetch(key, bases[t] + (size_t)experts[k] * slice);
        }
    }
}

void glm5_warm_slices(const void* gate_base, const void* up_base, const void* down_base,
                      int64_t slice, const int* experts, int n) {
#ifdef _WIN32
    if (!glm5_gpu_enabled() || slice <= 0) return;
    const uint8_t* bases[3] = { (const uint8_t*)gate_base, (const uint8_t*)up_base, (const uint8_t*)down_base };
    WIN32_MEMORY_RANGE_ENTRY entries[3 * 8];
    ULONG cnt = 0;
    for (int k = 0; k < n && k < 8; k++) {
        for (int t = 0; t < 3; t++) {
            entries[cnt].VirtualAddress = (PVOID)(bases[t] + (size_t)experts[k] * slice);
            entries[cnt].NumberOfBytes = (SIZE_T)slice;
            ++cnt;
        }
    }
    if (cnt) PrefetchVirtualMemory(GetCurrentProcess(), cnt, entries, 0);
#endif
}

} // namespace glm5
} // namespace nt

