#pragma once

#include <stdexcept>
#include <sys/resource.h>

// Process-wide counters include the worker threads. Take snapshots outside
// token timers; file output and repeat validation follow the second snapshot.
struct BenchmarkResources {
    long major_faults, minor_faults, input_blocks, output_blocks;
};

inline BenchmarkResources benchmark_resources() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
    return {usage.ru_majflt, usage.ru_minflt, usage.ru_inblock, usage.ru_oublock};
}

inline BenchmarkResources operator-(BenchmarkResources a, BenchmarkResources b) {
    return {a.major_faults - b.major_faults, a.minor_faults - b.minor_faults,
            a.input_blocks - b.input_blocks, a.output_blocks - b.output_blocks};
}
