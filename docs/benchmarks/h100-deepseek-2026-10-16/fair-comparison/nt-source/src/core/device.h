#pragma once

#include "types.h"
#include <string>

namespace nt {

// ============================================================
// GPU Device management
// 3 CUDA streams: compute, transfer0, transfer1
// For SLEP double-buffering in Phase 2
// ============================================================

struct GPUInfo {
    int     device_id;
    char    name[256];
    size_t  total_vram;
    size_t  free_vram;
    int     sm_count;
    int     compute_capability_major;
    int     compute_capability_minor;
    int     max_threads_per_block;
    int     warp_size;
    size_t  shared_mem_per_block;
    int     pcie_gen;      // PCIe generation
    int     pcie_width;    // PCIe lanes
};

enum StreamType {
    STREAM_COMPUTE   = 0,
    STREAM_TRANSFER0 = 1,  // For SLEP buffer A
    STREAM_TRANSFER1 = 2,  // For SLEP buffer B
    STREAM_COUNT     = 3,
};

class CUDADevice {
public:
    static CUDADevice& instance();

    bool init(int device_id = 0);
    void synchronize();
    void synchronize_stream(StreamType st);

    const GPUInfo& info() const { return info_; }

    // Stream access (opaque handle - cast to cudaStream_t in .cu files)
    void* stream(StreamType st) const { return streams_[st]; }

    // Event management for timing and sync
    void* create_event();
    void  destroy_event(void* event);
    void  record_event(void* event, StreamType st);
    void  wait_event(StreamType st, void* event);
    float elapsed_ms(void* start, void* end);

    // Memory info
    size_t free_vram() const;
    size_t total_vram() const;
    void   print_info() const;

    // Async memcpy on specific stream
    void memcpy_h2d_async(void* dst, const void* src, size_t size, StreamType st);
    void memcpy_d2h_async(void* dst, const void* src, size_t size, StreamType st);

private:
    CUDADevice() = default;
    ~CUDADevice();

    GPUInfo info_{};
    void*   streams_[STREAM_COUNT] = {};
    bool    initialized_ = false;
};

} // namespace nt

// ============================================================
// C linkage functions for tensor.cpp (compiled without nvcc)
// ============================================================
extern "C" {
    void* nt_cuda_malloc(size_t size);
    void  nt_cuda_free(void* ptr);
    void  nt_cuda_memcpy_h2d(void* dst, const void* src, size_t size);
    void  nt_cuda_memcpy_d2h(void* dst, const void* src, size_t size);
    void  nt_cuda_memcpy_d2d(void* dst, const void* src, size_t size);
    void  nt_cuda_memset(void* ptr, int value, size_t size);
    void* nt_cuda_malloc_host(size_t size);
    void  nt_cuda_free_host(void* ptr);
}
