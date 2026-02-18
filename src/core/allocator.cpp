#include "allocator.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// CUDA runtime stubs (linked from device.cu)
extern "C" {
    void* nt_cuda_malloc(size_t size);
    void  nt_cuda_free(void* ptr);
    void* nt_cuda_malloc_host(size_t size);
    void  nt_cuda_free_host(void* ptr);
}

namespace nt {

// ============================================================
// PoolAllocator
// ============================================================

PoolAllocator::PoolAllocator(Device device, size_t /*initial_pool_size*/)
    : device_(device)
{
}

PoolAllocator::~PoolAllocator() {
    for (auto& block : blocks_) {
        if (block.ptr) {
            raw_free(block.ptr);
        }
    }
    blocks_.clear();
}

void* PoolAllocator::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Round up to alignment
    size = (size + alignment - 1) & ~(alignment - 1);

    // Search for a free block of suitable size
    // Best-fit strategy to reduce fragmentation
    size_t best_idx = SIZE_MAX;
    size_t best_waste = SIZE_MAX;

    for (size_t i = 0; i < blocks_.size(); i++) {
        if (!blocks_[i].in_use && blocks_[i].size >= size) {
            size_t waste = blocks_[i].size - size;
            if (waste < best_waste) {
                best_waste = waste;
                best_idx = i;
                if (waste == 0) break;  // perfect fit
            }
        }
    }

    if (best_idx != SIZE_MAX) {
        blocks_[best_idx].in_use = true;
        total_in_use_ += blocks_[best_idx].size;
        peak_usage_ = std::max(peak_usage_, total_in_use_);
        return blocks_[best_idx].ptr;
    }

    // No suitable block found, allocate new
    void* ptr = raw_alloc(size);
    if (!ptr) return nullptr;

    PoolBlock block;
    block.ptr = ptr;
    block.size = size;
    block.in_use = true;

    size_t idx = blocks_.size();
    blocks_.push_back(block);
    ptr_to_block_[ptr] = idx;

    total_allocated_ += size;
    total_in_use_ += size;
    peak_usage_ = std::max(peak_usage_, total_in_use_);

    return ptr;
}

void PoolAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        fprintf(stderr, "PoolAllocator: deallocate unknown pointer %p\n", ptr);
        return;
    }

    size_t idx = it->second;
    blocks_[idx].in_use = false;
    total_in_use_ -= blocks_[idx].size;
}

void PoolAllocator::release_unused() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& block : blocks_) {
        if (!block.in_use && block.ptr) {
            ptr_to_block_.erase(block.ptr);
            total_allocated_ -= block.size;
            raw_free(block.ptr);
            block.ptr = nullptr;
            block.size = 0;
        }
    }

    // Compact the blocks vector
    blocks_.erase(
        std::remove_if(blocks_.begin(), blocks_.end(),
            [](const PoolBlock& b) { return b.ptr == nullptr; }),
        blocks_.end()
    );

    // Rebuild index
    ptr_to_block_.clear();
    for (size_t i = 0; i < blocks_.size(); i++) {
        ptr_to_block_[blocks_[i].ptr] = i;
    }
}

void* PoolAllocator::raw_alloc(size_t size) {
    if (device_ == Device::CUDA) {
        return nt_cuda_malloc(size);
    } else {
        return aligned_alloc(256, (size + 255) & ~255);
    }
}

void PoolAllocator::raw_free(void* ptr) {
    if (device_ == Device::CUDA) {
        nt_cuda_free(ptr);
    } else {
        ::free(ptr);
    }
}

// ============================================================
// Allocator (singleton)
// ============================================================

Allocator& Allocator::instance() {
    static Allocator alloc;
    return alloc;
}

Allocator::Allocator() {
    gpu_pool_ = std::make_unique<PoolAllocator>(Device::CUDA);
    cpu_pool_ = std::make_unique<PoolAllocator>(Device::CPU);
}

Allocator::~Allocator() = default;

void* Allocator::alloc(Device device, size_t size, size_t alignment) {
    if (device == Device::CUDA) {
        return gpu_pool_->allocate(size, alignment);
    } else {
        return cpu_pool_->allocate(size, alignment);
    }
}

void Allocator::free(Device device, void* ptr) {
    if (device == Device::CUDA) {
        gpu_pool_->deallocate(ptr);
    } else {
        cpu_pool_->deallocate(ptr);
    }
}

void* Allocator::alloc_pinned(size_t size) {
    return nt_cuda_malloc_host(size);
}

void Allocator::free_pinned(void* ptr) {
    nt_cuda_free_host(ptr);
}

void Allocator::print_stats() {
    fprintf(stderr, "=== Allocator Stats ===\n");
    fprintf(stderr, "GPU: allocated=%zuMB, in_use=%zuMB, peak=%zuMB\n",
        gpu_pool_->total_allocated() / (1024*1024),
        gpu_pool_->total_in_use() / (1024*1024),
        gpu_pool_->peak_usage() / (1024*1024));
    fprintf(stderr, "CPU: allocated=%zuMB, in_use=%zuMB, peak=%zuMB\n",
        cpu_pool_->total_allocated() / (1024*1024),
        cpu_pool_->total_in_use() / (1024*1024),
        cpu_pool_->peak_usage() / (1024*1024));
}

} // namespace nt
