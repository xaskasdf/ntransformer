#include "device.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace nt {

// ============================================================
// CUDADevice singleton
// ============================================================

CUDADevice& CUDADevice::instance() {
    static CUDADevice dev;
    return dev;
}

CUDADevice::~CUDADevice() {
    if (initialized_) {
        for (int i = 0; i < STREAM_COUNT; i++) {
            if (streams_[i]) {
                cudaStreamDestroy(static_cast<cudaStream_t>(streams_[i]));
            }
        }
    }
}

bool CUDADevice::init(int device_id) {
    if (initialized_) return true;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return false;
    }

    if (device_id >= device_count) {
        fprintf(stderr, "Device %d not found (have %d devices)\n", device_id, device_count);
        return false;
    }

    NT_CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    NT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    info_.device_id = device_id;
    strncpy(info_.name, prop.name, sizeof(info_.name) - 1);
    info_.total_vram = prop.totalGlobalMem;
    info_.sm_count = prop.multiProcessorCount;
    info_.compute_capability_major = prop.major;
    info_.compute_capability_minor = prop.minor;
    info_.max_threads_per_block = prop.maxThreadsPerBlock;
    info_.warp_size = prop.warpSize;
    info_.shared_mem_per_block = prop.sharedMemPerBlock;

    size_t free_mem, total_mem;
    NT_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info_.free_vram = free_mem;

    // Create streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStream_t s;
        NT_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        streams_[i] = static_cast<void*>(s);
    }

    initialized_ = true;
    return true;
}

void CUDADevice::synchronize() {
    NT_CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDADevice::synchronize_stream(StreamType st) {
    NT_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(streams_[st])));
}

void* CUDADevice::create_event() {
    cudaEvent_t event;
    NT_CUDA_CHECK(cudaEventCreate(&event));
    return static_cast<void*>(event);
}

void CUDADevice::destroy_event(void* event) {
    NT_CUDA_CHECK(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
}

void CUDADevice::record_event(void* event, StreamType st) {
    NT_CUDA_CHECK(cudaEventRecord(
        static_cast<cudaEvent_t>(event),
        static_cast<cudaStream_t>(streams_[st])
    ));
}

void CUDADevice::wait_event(StreamType st, void* event) {
    NT_CUDA_CHECK(cudaStreamWaitEvent(
        static_cast<cudaStream_t>(streams_[st]),
        static_cast<cudaEvent_t>(event), 0
    ));
}

float CUDADevice::elapsed_ms(void* start, void* end) {
    float ms = 0;
    NT_CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(end)));
    NT_CUDA_CHECK(cudaEventElapsedTime(&ms,
        static_cast<cudaEvent_t>(start),
        static_cast<cudaEvent_t>(end)
    ));
    return ms;
}

size_t CUDADevice::free_vram() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t CUDADevice::total_vram() const {
    return info_.total_vram;
}

void CUDADevice::print_info() const {
    fprintf(stderr, "=== GPU Info ===\n");
    fprintf(stderr, "Device: %s\n", info_.name);
    fprintf(stderr, "VRAM: %.1f GB (%.1f GB free)\n",
        info_.total_vram / (1024.0 * 1024 * 1024),
        info_.free_vram / (1024.0 * 1024 * 1024));
    fprintf(stderr, "SMs: %d, Compute: %d.%d\n",
        info_.sm_count, info_.compute_capability_major, info_.compute_capability_minor);
    fprintf(stderr, "Max threads/block: %d, Warp: %d\n",
        info_.max_threads_per_block, info_.warp_size);
    fprintf(stderr, "Shared mem/block: %zu KB\n", info_.shared_mem_per_block / 1024);
}

void CUDADevice::memcpy_h2d_async(void* dst, const void* src, size_t size, StreamType st) {
    NT_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(streams_[st])));
}

void CUDADevice::memcpy_d2h_async(void* dst, const void* src, size_t size, StreamType st) {
    NT_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
        static_cast<cudaStream_t>(streams_[st])));
}

} // namespace nt

// ============================================================
// C linkage CUDA operations (for tensor.cpp / allocator.cpp)
// ============================================================

extern "C" {

void* nt_cuda_malloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void nt_cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void nt_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void nt_cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void nt_cuda_memcpy_d2d(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void nt_cuda_memset(void* ptr, int value, size_t size) {
    cudaMemset(ptr, value, size);
}

void* nt_cuda_malloc_host(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost(%zu) failed: %s\n", size, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void nt_cuda_free_host(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

} // extern "C"
