#pragma once

#include "../core/device.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace nt {

// ============================================================
// CUDA event-based profiler
// Records timing of GPU operations using CUDA events
// ============================================================

struct ProfileEntry {
    std::string name;
    float ms;
    int count;
};

class Profiler {
public:
    static Profiler& instance();

    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    // Start/stop timing a named region
    void begin(const std::string& name, void* stream);
    void end(const std::string& name, void* stream);

    // Print summary
    void print_summary();
    void reset();

private:
    Profiler() = default;

    bool enabled_ = false;

    struct PendingEntry {
        void* start_event = nullptr;
        void* end_event = nullptr;
    };

    std::unordered_map<std::string, PendingEntry> pending_;
    std::unordered_map<std::string, std::pair<float, int>> totals_;  // total_ms, count
};

} // namespace nt
