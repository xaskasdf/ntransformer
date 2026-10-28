#include "model/ds41/ds41_cache.h"
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using nt::ds41::ExpertCacheIndex;
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void exercise(int percent) {
    ExpertCacheIndex cache;
    constexpr size_t capacity = 64;
    cache.configure(capacity, percent);
    std::vector<uint64_t> storage(capacity, UINT64_MAX);
    std::vector<uint64_t> lru_reference;
    std::mt19937 random(12345);
    for (int batch = 0; batch < 2000; ++batch) {
        cache.begin_batch();
        std::vector<std::pair<uint64_t, size_t>> selected;
        for (int i = 0; i < 24; ++i) {
            // Reused hot set, scans, duplicates and phase changes.
            const uint64_t key = batch % 13 == 0 ? 1000 + batch * 24 + i :
                (batch / 400) * 100 + random() % (i < 12 ? 16 : 100);
            auto result = cache.lookup_or_reserve(key);
            if (!percent) {
                auto it = std::find(lru_reference.begin(), lru_reference.end(), key);
                require(result.hit == (it != lru_reference.end()), "LRU differs from independent reference");
                if (it != lru_reference.end()) lru_reference.erase(it);
                lru_reference.push_back(key);
                if (lru_reference.size() > capacity) lru_reference.erase(lru_reference.begin());
            }
            require(result.slot < capacity, "slot out of bounds");
            if (result.hit) require(storage[result.slot] == key, "stale cache hit");
            else storage[result.slot] = key;
            selected.emplace_back(key, result.slot);
            for (auto [saved_key, slot] : selected)
                require(storage[slot] == saved_key, "selected slot evicted before completion");
        }
        require(cache.size() <= capacity && cache.protected_size() <= capacity * percent / 100,
            "capacity exceeded");
        if (batch == 999) { cache.clear(); lru_reference.clear(); std::fill(storage.begin(), storage.end(), UINT64_MAX); }
    }
    require(cache.stats().evictions > 0 && cache.stats().hits > 0, "missing replacement coverage");
    require((cache.stats().promotions > 0) == (percent > 0), "wrong cache policy");
}
void retention() {
    for (int percent : {0, 80}) {
        ExpertCacheIndex c; c.configure(10, percent);
        for (int pass = 0; pass < 2; ++pass) {
            c.begin_batch(); for (int k = 0; k < 6; ++k) c.lookup_or_reserve(k);
        }
        for (int k = 100; k < 120; ++k) { c.begin_batch(); c.lookup_or_reserve(k); }
        c.begin_batch();
        for (int k = 0; k < 6; ++k)
            require(c.lookup_or_reserve(k).hit == (percent != 0), "scan retention mismatch");
    }
    ExpertCacheIndex c; c.configure(2, 50); c.begin_batch();
    c.lookup_or_reserve(1); c.lookup_or_reserve(1); c.lookup_or_reserve(2);
    require(c.stats().promotions == 0, "same-operation duplicate promoted");
    bool rejected = false;
    try { c.lookup_or_reserve(3); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && c.size() == 2, "all-reserved failure changed cache");
    require(c.lookup_or_reserve(1).hit && c.lookup_or_reserve(2).hit, "all-reserved failure lost entries");
    c.begin_batch(); require(!c.lookup_or_reserve(3).hit, "old reservation not released");
}
}
int main() {
    try {
        for (int percent : {0, 50, 80, 90, 99}) exercise(percent);
        retention();
        std::puts("PASS cache policy: reservations, duplicate keys, scans, reuse, demotion, phase changes and invalidation");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
