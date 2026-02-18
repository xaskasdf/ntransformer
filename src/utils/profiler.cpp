#include "profiler.h"
#include <cstdio>
#include <algorithm>

namespace nt {

Profiler& Profiler::instance() {
    static Profiler prof;
    return prof;
}

void Profiler::begin(const std::string& name, void* stream) {
    if (!enabled_) return;

    auto& dev = CUDADevice::instance();
    PendingEntry entry;
    entry.start_event = dev.create_event();
    entry.end_event = dev.create_event();

    StreamType st = STREAM_COMPUTE;  // default to compute stream
    dev.record_event(entry.start_event, st);

    pending_[name] = entry;
}

void Profiler::end(const std::string& name, void* stream) {
    if (!enabled_) return;

    auto it = pending_.find(name);
    if (it == pending_.end()) return;

    auto& dev = CUDADevice::instance();
    dev.record_event(it->second.end_event, STREAM_COMPUTE);

    float ms = dev.elapsed_ms(it->second.start_event, it->second.end_event);

    auto& total = totals_[name];
    total.first += ms;
    total.second++;

    dev.destroy_event(it->second.start_event);
    dev.destroy_event(it->second.end_event);
    pending_.erase(it);
}

void Profiler::print_summary() {
    if (totals_.empty()) {
        fprintf(stderr, "Profiler: no data\n");
        return;
    }

    // Sort by total time descending
    std::vector<std::pair<std::string, std::pair<float, int>>> entries(totals_.begin(), totals_.end());
    std::sort(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) { return a.second.first > b.second.first; });

    fprintf(stderr, "=== Profiler Summary ===\n");
    fprintf(stderr, "%-30s %10s %8s %10s\n", "Name", "Total(ms)", "Count", "Avg(ms)");
    fprintf(stderr, "%-30s %10s %8s %10s\n", "----", "--------", "-----", "------");

    for (const auto& [name, data] : entries) {
        fprintf(stderr, "%-30s %10.2f %8d %10.2f\n",
            name.c_str(), data.first, data.second, data.first / data.second);
    }
}

void Profiler::reset() {
    for (auto& [name, entry] : pending_) {
        CUDADevice::instance().destroy_event(entry.start_event);
        CUDADevice::instance().destroy_event(entry.end_event);
    }
    pending_.clear();
    totals_.clear();
}

} // namespace nt
