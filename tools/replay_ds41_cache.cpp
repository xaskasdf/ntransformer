// Offline replay of an ordered per-owner trace. No CUDA or model files needed.
#include "model/ds41/ds41_cache.h"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: protected_percent ordered-trace-files...");
        const int percent = std::stoi(argv[1]);
        nt::ds41::ExpertCacheIndex cache;
        std::unordered_set<uint64_t> seen;
        uint64_t previous_batch = 0, capacity = 0, slice = 0;
        for (int file = 2; file < argc; ++file) {
            std::ifstream input(argv[file], std::ios::binary);
            uint64_t header[4]{};
            input.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!input || header[0] != 0x4e54434143484531ull || !header[1] || !header[2])
                throw std::runtime_error("invalid cache trace header");
            if (file == 2) { capacity = header[1]; slice = header[2]; cache.configure(capacity, percent); }
            if (capacity != header[1] || slice != header[2]) throw std::runtime_error("trace capacity changed");
            const auto before = cache.stats();
            uint64_t first_use = 0, recorded_misses = 0, hit_mismatches = 0;
            for (uint64_t i = 0; i < header[3]; ++i) {
                uint64_t record[3]{};
                input.read(reinterpret_cast<char*>(record), sizeof(record));
                if (!input || !record[1] || record[1] < previous_batch || record[2] > 1)
                    throw std::runtime_error("invalid or nonmonotonic trace record");
                if (record[1] != previous_batch) { cache.begin_batch(); previous_batch = record[1]; }
                first_use += seen.insert(record[0]).second;
                recorded_misses += !record[2];
                const auto result = cache.lookup_or_reserve(record[0]);
                hit_mismatches += result.hit != bool(record[2]);
            }
            if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing trace bytes");
            const auto after = cache.stats();
            const uint64_t misses = after.misses - before.misses;
            std::cout << "{\"file\":" << std::quoted(argv[file]) << ",\"protected_percent\":" << percent
                << ",\"capacity\":" << capacity << ",\"slice_bytes\":" << slice << ",\"lookups\":" << header[3]
                << ",\"hits\":" << after.hits - before.hits << ",\"misses\":" << misses
                << ",\"first_use\":" << first_use << ",\"recurrent_misses\":" << misses - first_use
                << ",\"recorded_misses\":" << recorded_misses << ",\"hit_mismatches\":" << hit_mismatches
                << ",\"evictions\":" << after.evictions - before.evictions
                << ",\"promotions\":" << after.promotions - before.promotions
                << ",\"demotions\":" << after.demotions - before.demotions << "}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
