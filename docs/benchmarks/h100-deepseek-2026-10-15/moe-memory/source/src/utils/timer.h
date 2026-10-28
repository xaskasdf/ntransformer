#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace nt {

// ============================================================
// Simple CPU timer utility
// ============================================================
class Timer {
public:
    Timer() { reset(); }

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    float elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(now - start_).count();
    }

    float elapsed_s() const { return elapsed_ms() / 1000.0f; }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// RAII scoped timer that prints on destruction
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name) {}

    ~ScopedTimer() {
        fprintf(stderr, "[%s] %.2f ms\n", name_.c_str(), timer_.elapsed_ms());
    }

private:
    std::string name_;
    Timer timer_;
};

} // namespace nt
