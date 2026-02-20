#pragma once

#include "types.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace nt {

// ============================================================
// Pool allocator for VRAM and pinned host RAM
// Reduces cudaMalloc/cudaFree overhead during inference
// ============================================================

struct PoolBlock {
    void*  ptr;
    size_t size;
    bool   in_use;
};

class PoolAllocator {
public:
    explicit PoolAllocator(Device device, size_t initial_pool_size = 0);
    ~PoolAllocator();

    // Non-copyable
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    void* allocate(size_t size, size_t alignment = 256);
    void  deallocate(void* ptr);

    // Release all unused blocks back to the system
    void release_unused();

    // Stats
    size_t total_allocated() const { return total_allocated_; }
    size_t total_in_use() const { return total_in_use_; }
    size_t peak_usage() const { return peak_usage_; }

    Device device() const { return device_; }

private:
    Device device_;
    std::vector<PoolBlock> blocks_;
    std::unordered_map<void*, size_t> ptr_to_block_;  // ptr -> block index
    std::mutex mutex_;

    size_t total_allocated_ = 0;
    size_t total_in_use_ = 0;
    size_t peak_usage_ = 0;

    void* raw_alloc(size_t size);
    void  raw_free(void* ptr);
};

// ============================================================
// Global allocator access
// ============================================================
class Allocator {
public:
    static Allocator& instance();

    void* alloc(Device device, size_t size, size_t alignment = 256);
    void  free(Device device, void* ptr);

    // Pinned (page-locked) host memory for async transfers
    void* alloc_pinned(size_t size);
    void  free_pinned(void* ptr);

    PoolAllocator& gpu_pool() { return *gpu_pool_; }
    PoolAllocator& cpu_pool() { return *cpu_pool_; }

    void print_stats();

private:
    Allocator();
    ~Allocator();

    std::unique_ptr<PoolAllocator> gpu_pool_;
    std::unique_ptr<PoolAllocator> cpu_pool_;
};

} // namespace nt
