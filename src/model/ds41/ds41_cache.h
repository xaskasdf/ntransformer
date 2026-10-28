#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace nt::ds41 {

// Slot metadata only: this class neither allocates nor transfers GPU weights.
// The caller begins one batch after the previous batch has finished and keeps
// all returned slots alive until its streams drain. Repeated keys share a slot.
class ExpertCacheIndex {
public:
    struct Result { size_t slot; bool hit; };
    struct Stats {
        uint64_t hits = 0, misses = 0, evictions = 0;
        uint64_t promotions = 0, demotions = 0;
    };
    ExpertCacheIndex() = default;
    ExpertCacheIndex(const ExpertCacheIndex&) = delete;
    ExpertCacheIndex& operator=(const ExpertCacheIndex&) = delete;

    void configure(size_t capacity, int protected_percent = 0) {
        if (!capacity || protected_percent < 0 || protected_percent >= 100)
            throw std::invalid_argument("invalid expert cache capacity/policy");
        clear();
        capacity_ = capacity;
        protected_limit_ = capacity * protected_percent / 100;
        stats_ = {};
        entries_.reserve(capacity);
    }
    void clear() {
        entries_.clear(); probation_.clear(); protected_.clear();
        next_slot_ = 0; epoch_ = 0;
    }
    void begin_batch() {
        if (++epoch_ == 0) {
            for (auto& [key, entry] : entries_) entry.reserved = 0;
            epoch_ = 1;
        }
    }
    Result lookup_or_reserve(uint64_t key) {
        if (!epoch_ || !capacity_) throw std::logic_error("expert cache batch not started");
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            auto& entry = found->second;
            ++stats_.hits;
            // A duplicate in one operation is not evidence of temporal reuse.
            const bool reused = entry.reserved != epoch_;
            entry.reserved = epoch_;
            if (entry.protected_entry) protected_.splice(protected_.begin(), protected_, entry.position);
            else if (protected_limit_ && reused) {
                protected_.splice(protected_.begin(), probation_, entry.position);
                entry.protected_entry = true;
                ++stats_.promotions;
                if (protected_.size() > protected_limit_) {
                    const uint64_t demoted = protected_.back();
                    auto& other = entries_.at(demoted);
                    probation_.splice(probation_.begin(), protected_, other.position);
                    other.protected_entry = false;
                    ++stats_.demotions;
                }
            } else probation_.splice(probation_.begin(), probation_, entry.position);
            return {entry.slot, true};
        }
        size_t slot;
        if (next_slot_ < capacity_) slot = next_slot_++;
        else {
            // Reservation, rather than list position, guarantees safety when a
            // selected protected entry is demoted during this same operation.
            auto* queue = &probation_;
            auto victim = unreserved(*queue);
            if (victim == queue->end()) { queue = &protected_; victim = unreserved(*queue); }
            if (victim == queue->end()) throw std::runtime_error("all expert cache slots are reserved");
            auto it = entries_.find(*victim);
            slot = it->second.slot;
            entries_.erase(it); queue->erase(victim);
            ++stats_.evictions;
        }
        probation_.push_front(key);
        entries_.emplace(key, Entry{slot, probation_.begin(), epoch_, false});
        ++stats_.misses;
        return {slot, false};
    }
    const Stats& stats() const { return stats_; }
    size_t capacity() const { return capacity_; }
    size_t size() const { return entries_.size(); }
    size_t protected_size() const { return protected_.size(); }
    uint64_t epoch() const { return epoch_; }

    // read-only membership test; does not touch LRU positions
    bool contains(uint64_t key) const { return entries_.count(key) > 0; }

private:
    using Queue = std::list<uint64_t>;
    struct Entry {
        size_t slot;
        Queue::iterator position;
        uint64_t reserved;
        bool protected_entry;
    };
    Queue::iterator unreserved(Queue& queue) {
        auto it = queue.end();
        while (it != queue.begin()) {
            --it;
            if (entries_.at(*it).reserved != epoch_) return it;
        }
        return queue.end();
    }
    size_t capacity_ = 0, protected_limit_ = 0, next_slot_ = 0;
    uint64_t epoch_ = 0;
    Queue probation_, protected_;
    std::unordered_map<uint64_t, Entry> entries_;
    Stats stats_;
};
} // namespace nt::ds41
