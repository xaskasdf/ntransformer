#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nt {

// Fixed-size weight blocks owned by a second CUDA device. Compute and staging
// stay on the primary device; transfers use its non-default CUDA streams.
class PeerCache {
public:
    PeerCache() = default;
    ~PeerCache();
    PeerCache(const PeerCache&) = delete;
    PeerCache& operator=(const PeerCache&) = delete;

    bool init(int compute_device, int peer_device, size_t block_bytes,
              size_t reserve_bytes = 2ULL << 30);
    int allocate(int requested_blocks);
    bool upload(int block, size_t offset, const void* source, size_t bytes);
    bool copy_to_compute(int block, void* destination, size_t bytes, void* stream);
    void shutdown();

    int device_id() const { return peer_device_; }
    int block_count() const { return static_cast<int>(blocks_.size()); }
    size_t block_bytes() const { return block_bytes_; }
    size_t allocated_bytes() const { return blocks_.size() * block_bytes_; }
    uint64_t transfer_count() const { return transfer_count_; }
    uint64_t transferred_bytes() const { return transferred_bytes_; }

private:
    int compute_device_ = -1;
    int peer_device_ = -1;
    size_t block_bytes_ = 0;
    size_t available_bytes_ = 0;
    std::vector<void*> blocks_;
    uint64_t transfer_count_ = 0;
    uint64_t transferred_bytes_ = 0;
};

} // namespace nt
