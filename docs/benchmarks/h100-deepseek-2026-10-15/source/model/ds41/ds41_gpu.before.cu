// Expert offload for DeepSeek-V4.1 MXFP4 expert tensors.
//
// One CUDA device is used as an LRU cache for expert weight slices
// (gate/up/down, 6.27 MiB each in the real model) plus a small compute
// pipeline: GEMV (MXFP4) -> swiglu-clamp -> GEMV (down) -> weighted add.

#include "ds41_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
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

__constant__ float c_kvals[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
};

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

// y[row] = W[row, :] @ x   (W: rows x cols, MXFP4 blocks of 32)
__global__ void k_gemv_mxfp4(const uint8_t* __restrict__ W,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;

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
            acc += d * (c_kvals[qs[j] & 15] * xb[j] + c_kvals[qs[j] >> 4] * xb[j + 16]);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    }
    if (lane == 0) y[wid] = acc;
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
        char* p = nullptr;
        size_t scratch = 0;
        // scratch buffers grow with the model; allocate generously for deepseek41
        scratch = 8192 * sizeof(float);
        if (!alloc((void**)&x_buf_, scratch)) { return false; }
        if (!alloc((void**)&gate_buf_, 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&up_buf_, 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&h_buf_, 4096 * sizeof(float))) { return false; }
        if (!alloc((void**)&down_buf_, 8192 * sizeof(float))) { return false; }
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
    ExpertCache& c = cache();
    const size_t row_bytes_gate = (size_t)(hidden / 32) * 17;
    const size_t row_bytes_down = (size_t)(ff / 32) * 17;
    const size_t gate_slice = (size_t)ff * row_bytes_gate;
    const size_t down_slice = (size_t)hidden * row_bytes_down;
    const size_t slice_bytes = gate_slice > down_slice ? gate_slice : down_slice;

    if (!c.ensure_init(slice_bytes) || !c.available()) return false;
    if (n_used > 8) return false;

    cudaStream_t s = c.stream();

    cudaMemcpyAsync(c.x_buf(), x, hidden * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemsetAsync(c.out_buf(), 0, hidden * sizeof(float), s);

    struct Miss { size_t idx; const void* host; void* dst; size_t bytes; };
    std::vector<Miss> misses;
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

    // stage misses into pinned memory in parallel, then enqueue async H2D copies
    if (!misses.empty()) {
        const int npin = c.pin_ring_size();
        std::vector<std::thread> copiers;
        copiers.reserve(misses.size());
        for (size_t i = 0; i < misses.size(); i++) {
            copiers.emplace_back([&, i]() {
                memcpy(c.pin_ring((int)(i % npin)), misses[i].host, misses[i].bytes);
            });
        }
        for (auto& th : copiers) th.join();
        for (size_t i = 0; i < misses.size(); i++) {
            cudaMemcpyAsync(misses[i].dst, c.pin_ring((int)(i % npin)), misses[i].bytes,
                            cudaMemcpyHostToDevice, s);
        }
    }

    for (int k = 0; k < n_used; k++) {
        const int rows_gate = (int)ff;
        const int cols_gate = (int)hidden;
        const int threads_gate = rows_gate * 32;
        k_gemv_mxfp4<<<(threads_gate + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            static_cast<const uint8_t*>(dev_gate[k]), c.x_buf(), c.gate_buf(), rows_gate, cols_gate);
        k_gemv_mxfp4<<<(threads_gate + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            static_cast<const uint8_t*>(dev_up[k]), c.x_buf(), c.up_buf(), rows_gate, cols_gate);

        k_silu_clamp<<<(rows_gate + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            c.gate_buf(), c.up_buf(), c.h_buf(), rows_gate, limit);

        const int rows_down = (int)hidden;
        const int cols_down = (int)ff;
        const int threads_down = rows_down * 32;
        k_gemv_mxfp4<<<(threads_down + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            static_cast<const uint8_t*>(dev_down[k]), c.h_buf(), c.down_buf(), rows_down, cols_down);

        k_axpy<<<(rows_down + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            c.out_buf(), c.down_buf(), weights[k], rows_down);
    }

    if (!d2h_stage(out, c.out_buf(), hidden * sizeof(float), s)) {
        fprintf(stderr, "ds41-gpu: moe output copy failed\n");
        return false;
    }
    return true;
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
    const int threads = (int)rows * 32;
    k_gemv_mxfp4<<<(threads + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        static_cast<const uint8_t*>(dev), c.x_buf(), c.down_buf(), (int)rows, (int)cols);
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
__global__ void k_gemv_q8_0(const uint8_t* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int nb = cols >> 5;
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

const void* q8_row_bytes_ok(const void* w, int64_t rows, int64_t cols, size_t* bytes) {
    *bytes = (size_t)rows * (size_t)(cols / 32) * 34u;
    return w;
}

bool gpu_backbone_enabled() {
    return cache().ensure_device();
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
                       float* qr_out, float* q_out, float* kv_out) {
    ExpertCache& c = cache();
    if (!c.ensure_device()) return false;
    cudaStream_t s = c.stream();

    auto gt = GT0();
    size_t bytes;
    const void* d_wq_a = g_resident.get(q8_row_bytes_ok(wq_a, q_lora, hidden, &bytes), bytes);
    const void* d_wq_b = g_resident.get(q8_row_bytes_ok(wq_b, n_heads * head_dim, q_lora, &bytes), bytes);
    const void* d_wkv  = g_resident.get(q8_row_bytes_ok(wkv, head_dim, hidden, &bytes), bytes);
    const void* d_q_norm  = g_resident.get(q_a_norm, (size_t)q_lora * 4);
    const void* d_kv_norm = g_resident.get(kv_a_norm, (size_t)head_dim * 4);
    if (!d_wq_a || !d_wq_b || !d_wkv || !d_q_norm || !d_kv_norm) { if (g_gputime) fprintf(stderr, "[prologue] resident fail a=%p b=%p kv=%p qn=%p kvn=%p\n", d_wq_a, d_wq_b, d_wkv, d_q_norm, d_kv_norm); return false; }

    GT_ACC(g_t_res, gt);
    gt = GT0();
    float* d_qr = scratch_slot(g_attn.qr, g_attn.qr_n, q_lora);
    float* d_q  = scratch_slot(g_attn.q,  g_attn.q_n,  n_heads * head_dim);
    float* d_kv = scratch_slot(g_attn.kv, g_attn.kv_n, head_dim);
    if (!d_qr || !d_q || !d_kv) { if (g_gputime) fprintf(stderr, "[prologue] scratch fail\n"); return false; }

    // rope tables (static device buffers)
    const int half = (int)(rope_dim / 2);
    float* d_cos = rope_dev_cos();
    float* d_sin = rope_dev_sin();
    if (!d_cos || !d_sin) { if (g_gputime) fprintf(stderr, "[prologue] rope buffers fail\n"); return false; }
    cudaMemcpyAsync(d_cos, cos_t, half * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_sin, sin_t, half * 4, cudaMemcpyHostToDevice, s);

    h2d_stage(c.x_buf(), x, hidden * 4, s);

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
    // note: d2h_stage syncs; ordering of the three copies is preserved by the stream
    if (!d2h_stage(qr_out, d_qr, q_lora * 4, s) ||
        !d2h_stage(q_out, d_q, (size_t)rows_q * 4, s) ||
        !d2h_stage(kv_out, d_kv, head_dim * 4, s)) {
        fprintf(stderr, "ds41-gpu: attn prologue copy failed\n");
        return false;
    }
    GT_ACC(g_t_copy, gt);
    GT_MAYBE_PRINT();
    return true;
}

bool gpu_attn_tail(const float* attn_out, int64_t head_dim, int64_t n_heads,
                   const void* wo_a, const void* wo_b,
                   int64_t o_groups, int64_t o_lora, int64_t hidden,
                   float* out) {
    ExpertCache& c = cache();
    if (!c.ensure_device()) return false;
    cudaStream_t s = c.stream();

    const int64_t gdim = n_heads / o_groups * head_dim;
    size_t bytes;
    const void* d_wo_a = g_resident.get(q8_row_bytes_ok(wo_a, o_groups * o_lora, gdim, &bytes), bytes);
    const void* d_wo_b = g_resident.get(q8_row_bytes_ok(wo_b, hidden, o_groups * o_lora, &bytes), bytes);
    if (!d_wo_a || !d_wo_b) return false;

    float* d_in = scratch_slot(g_attn.q, g_attn.q_n, n_heads * head_dim);
    float* d_oa = scratch_slot(g_attn.oa, g_attn.oa_n, o_groups * o_lora);
    float* d_out = scratch_slot(g_attn.out, g_attn.out_n, hidden);
    if (!d_in || !d_oa || !d_out) return false;

    h2d_stage(d_in, attn_out, (size_t)n_heads * head_dim * 4, s);

    const size_t a_row = (size_t)(gdim / 32) * 34u;
    for (int64_t gg = 0; gg < o_groups; gg++) {
        const uint8_t* wg = (const uint8_t*)d_wo_a + (size_t)gg * o_lora * a_row;
        const float* xg = d_in + (size_t)gg * gdim;
        float* yg = d_oa + (size_t)gg * o_lora;
        k_gemv_q8_0<<<(int)(o_lora * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
            wg, xg, yg, (int)o_lora, (int)gdim);
    }
    k_gemv_q8_0<<<(int)(hidden * 32 + kThreads - 1) / kThreads, kThreads, 0, s>>>(
        (const uint8_t*)d_wo_b, d_oa, d_out, (int)hidden, (int)(o_groups * o_lora));
    return d2h_stage(out, d_out, (size_t)hidden * 4, s);
}

bool gpu_comp_state(const float* x, int64_t hidden, int64_t latent,
                    const void* comp_wkv, const void* comp_gate,
                    float* skv_out, float* sscore_out) {
    ExpertCache& c = cache();
    if (!c.ensure_device()) return false;
    cudaStream_t s = c.stream();
    const int64_t hd = latent;

    const void* d_wkv_g = g_resident.get(comp_wkv, (size_t)hidden * hd * 2);
    if (!d_wkv_g) return false;
    const void* d_gate = comp_gate ? g_resident.get(comp_gate, (size_t)hidden * hd * 2) : nullptr;
    if (comp_gate && !d_gate) return false;

    float* d_skv = scratch_slot(g_attn.qr, g_attn.qr_n, hd);
    float* d_sco = sscore_out ? scratch_slot(g_attn.kv, g_attn.kv_n, hd) : nullptr;
    if (!d_skv || (sscore_out && !d_sco)) return false;

    h2d_stage(c.x_buf(), x, hidden * 4, s);
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
    if (!c.ensure_device()) return false;
    cudaStream_t s = c.stream();

    float* d_out = scratch_slot(g_attn.oa, g_attn.oa_n, out_rows);
    if (!d_out) return false;
    h2d_stage(c.x_buf(), x, hidden_in * 4, s);

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
