#include "peer_cache.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

namespace nt {
namespace {

bool check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    fprintf(stderr, "PeerCache: %s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

class DeviceScope {
public:
    explicit DeviceScope(int device) {
        valid_ = check_cuda(cudaGetDevice(&previous_), "get current device") &&
                 check_cuda(cudaSetDevice(device), "select device");
    }
    ~DeviceScope() {
        if (previous_ >= 0) cudaSetDevice(previous_);
    }
    bool valid() const { return valid_; }

private:
    int previous_ = -1;
    bool valid_ = false;
};

} // namespace

PeerCache::~PeerCache() {
    shutdown();
}

bool PeerCache::init(int compute_device, int peer_device, size_t block_bytes,
                     size_t reserve_bytes) {
    shutdown();
    int count = 0;
    if (!check_cuda(cudaGetDeviceCount(&count), "enumerate devices")) return false;
    if (compute_device < 0 || peer_device < 0 || compute_device >= count ||
        peer_device >= count || compute_device == peer_device || block_bytes == 0) {
        fprintf(stderr, "PeerCache: requires two distinct visible GPUs and nonempty blocks\n");
        return false;
    }
    int capable = 0;
    if (!check_cuda(cudaDeviceCanAccessPeer(&capable, compute_device, peer_device),
                    "query P2P access") || !capable) {
        fprintf(stderr, "PeerCache: direct GPU %d -> GPU %d access unavailable\n",
                compute_device, peer_device);
        return false;
    }
    {
        DeviceScope compute(compute_device);
        if (!compute.valid()) return false;
        auto error = cudaDeviceEnablePeerAccess(peer_device, 0);
        if (error == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        } else if (!check_cuda(error, "enable P2P access")) {
            return false;
        }
    }
    {
        DeviceScope peer(peer_device);
        if (!peer.valid()) return false;
        size_t free_bytes = 0, total_bytes = 0;
        if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query peer VRAM")) return false;
        available_bytes_ = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    }
    compute_device_ = compute_device;
    peer_device_ = peer_device;
    block_bytes_ = block_bytes;
    fprintf(stderr, "PeerCache: GPU %d cache -> GPU %d compute, %.1f MiB/block, "
            "%.1f GiB available after reserve\n", peer_device_, compute_device_,
            block_bytes_ / (1024.0 * 1024), available_bytes_ / (1024.0 * 1024 * 1024));
    return true;
}

int PeerCache::allocate(int requested_blocks) {
    if (peer_device_ < 0 || requested_blocks <= 0) return block_count();
    DeviceScope peer(peer_device_);
    if (!peer.valid()) return block_count();
    const size_t target = std::min(static_cast<size_t>(requested_blocks), available_bytes_ / block_bytes_);
    while (blocks_.size() < target) {
        void* block = nullptr;
        auto error = cudaMalloc(&block, block_bytes_);
        if (error != cudaSuccess) {
            check_cuda(error, "allocate peer block (remaining layers use host cache)");
            cudaGetLastError();
            break;
        }
        if (!check_cuda(cudaMemset(block, 0, block_bytes_), "initialize peer block")) {
            cudaFree(block);
            break;
        }
        blocks_.push_back(block);
    }
    return block_count();
}

bool PeerCache::upload(int block, size_t offset, const void* source, size_t bytes) {
    if (block < 0 || block >= block_count() || !source || offset > block_bytes_ ||
        bytes > block_bytes_ - offset) return false;
    DeviceScope peer(peer_device_);
    if (!peer.valid()) return false;
    auto* destination = static_cast<unsigned char*>(blocks_[block]) + offset;
    return check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                      "upload peer weights");
}

bool PeerCache::copy_to_compute(int block, void* destination, size_t bytes, void* stream) {
    if (block < 0 || block >= block_count() || !destination || bytes > block_bytes_ || !stream) return false;
    DeviceScope compute(compute_device_);
    if (!compute.valid()) return false;
    if (!check_cuda(cudaMemcpyPeerAsync(destination, compute_device_, blocks_[block],
                    peer_device_, bytes, static_cast<cudaStream_t>(stream)), "async P2P copy")) return false;
    ++transfer_count_;
    transferred_bytes_ += bytes;
    return true;
}

void PeerCache::shutdown() {
    if (!blocks_.empty()) {
        // Wait for this process's readers before freeing their peer allocations.
        DeviceScope compute(compute_device_);
        if (compute.valid()) cudaDeviceSynchronize();
        DeviceScope peer(peer_device_);
        if (peer.valid()) {
            for (void* block : blocks_) cudaFree(block);
        }
    }
    blocks_.clear();
    compute_device_ = peer_device_ = -1;
    block_bytes_ = available_bytes_ = 0;
    transfer_count_ = transferred_bytes_ = 0;
    // Access is a property of CUDA primary contexts. Keep it enabled for other
    // users in this process instead of disabling mappings that they may share.
}

} // namespace nt
