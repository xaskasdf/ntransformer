// Scoped host work must not expose an unfinished GPU result or outlive a call.
#include "model/ds41/ds41_gpu.h"
#include "model/ds41/ds41_model.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Fixture {
    static constexpr int count = 8;
    int hidden, ff;
    std::vector<uint8_t> gate, up, down;
    std::vector<float> x;
    int expected_groups = 0, expected_remote = 0, expected_remote_calls = 0;
    int ids[count] = {7, 0, 5, 2, 6, 1, 4, 3};
    float weights[count] = {0.03f, 0.2f, 0.17f, 0.1f, 0.09f, 0.15f, 0.12f, 0.14f};
    Fixture(int h, int f) : hidden(h), ff(f), gate(count * ff * hidden / 32 * 17),
                            up(gate.size()), down(gate.size()), x(hidden) {
        int seed = 0;
        for (auto* matrix : {&gate, &up, &down}) {
            ++seed;
            for (size_t b = 0; b < matrix->size() / 17; ++b) {
                (*matrix)[b * 17] = 120 + b % 3;
                for (int i = 1; i < 17; ++i) (*matrix)[b * 17 + i] = uint8_t(b * 19 + i * 11 + seed);
            }
        }
    }
    bool run(std::vector<float>& out, int used, const std::function<void()>& work = {}) {
        const bool ok = nt::ds41::gpu_moe_experts(x.data(), hidden, gate.data(), up.data(), down.data(),
                                        ff, ids, weights, used, 2.0f, out.data(), work);
        if (ok) {
            int remote = 0;
            for (int i = 0; i < used; ++i) remote += ids[i] & 1;
            expected_groups += int(remote > 0) + int(remote < used);
            expected_remote += remote;
            expected_remote_calls += remote > 0;
            int device = -1;
            require(cudaGetDevice(&device) == cudaSuccess && device == 0, "peer operation did not restore primary device");
        }
        return ok;
    }
};
}

int main(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "small";
        require(mode == "small" || mode == "real" || mode == "max" || mode == "disabled", "unknown shape");
        Fixture f(mode == "max" ? 8192 : mode == "real" ? 5120 : 256,
                  mode == "max" ? 4096 : mode == "real" ? 2304 : 128);
        std::vector<float> out(f.hidden, -123.0f);
        std::vector<float> dump;
        int calls = 0;
        if (argc > 1 && std::string(argv[1]) == "disabled") {
            require(!f.run(out, 6, [&] { ++calls; }), "disabled GPU path succeeded");
            require(calls == 0, "disabled path ran callback");
            for (float v : out) require(v == -123.0f, "disabled path published output");
            std::puts("PASS disabled GPU leaves callback and output untouched");
            return 0;
        }
        if (!nt::ds41::gpu_experts_enabled()) {
            std::puts("SKIP CUDA unavailable or DS41_GPU=0"); return 77;
        }
        const std::string peer_mode = std::getenv("DS41_PEER_MODE") ? std::getenv("DS41_PEER_MODE") : "0";
        const bool peer = peer_mode == "copy" || peer_mode == "execute";
        const bool execute = peer_mode == "execute";
        if (peer) {
            int count = 0, capable = 0;
            cudaGetDeviceCount(&count);
            if (count >= 2) cudaDeviceCanAccessPeer(&capable, 0, 1);
            if (!capable) { std::puts("SKIP two P2P CUDA devices required"); return 77; }
        }
        if (std::getenv("DS41_TEST_LOCAL_ONLY")) for (int& id : f.ids) id &= ~1;
        const auto caller = std::this_thread::get_id();
        for (int round = 0; round < 3; ++round) {
            if (round == 2) {
                f.ids[7] = f.ids[0]; // duplicate routing must retain its own weight
                f.weights[1] = -0.2f;
                f.weights[3] = -0.0f;
            }
            for (int i = 0; i < f.hidden; ++i) f.x[i] = std::sin(float(i + 1) * (round + 1) * 0.07f);
            for (int used : {1, 6, 8}) {
                std::vector<float> reference(f.hidden);
                std::fill(out.begin(), out.end(), -123.0f);
                std::vector<float> cpu(257);
                calls = 0;
                require(f.run(out, used, [&] {
                    ++calls;
                    require(std::this_thread::get_id() == caller, "host work changed calling thread");
                    for (float v : out) require(v == -123.0f, "GPU output published before callback completed");
                    nt::ds41::parallel_for(cpu.size(), 0, [&](int64_t i, int) { cpu[i] = float(i) * 0.125f; });
                }), "overlap unavailable");
                require(calls == 1, "callback not executed exactly once");
                // Candidate runs first to exercise cold uploads as well as hits.
                require(f.run(reference, used), "synchronous reference unavailable");
                for (int i = 0; i < (int)cpu.size(); ++i) require(cpu[i] == float(i) * 0.125f, "CPU work incomplete");
                for (float v : out) require(std::isfinite(v), "nonfinite output");
                require(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0,
                        "overlap changed routed expert outputs");
                dump.insert(dump.end(), out.begin(), out.end());
            }
        }
        for (int used : {0, 9}) {
            calls = 0;
            require(!f.run(out, used, [&] { ++calls; }) && calls == 0, "invalid count ran callback");
        }
        std::vector<float> recovery_reference(f.hidden);
        require(f.run(recovery_reference, 6), "exception reference unavailable");
        std::fill(out.begin(), out.end(), -123.0f);
        bool caught = false;
        try {
            f.run(out, 6, [] { throw std::runtime_error("expected callback exception"); });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "expected callback exception";
        }
        require(caught, "callback exception not propagated");
        for (float v : out) require(v == -123.0f, "exception published GPU output");
        require(cudaDeviceSynchronize() == cudaSuccess, "callback exception left CUDA error");
        require(f.run(out, 6), "GPU cache not reusable after callback exception");
        require(std::memcmp(out.data(), recovery_reference.data(), out.size() * sizeof(float)) == 0,
                "exception recovery changed outputs");
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution error");
        auto stats = nt::ds41::gpu_moe_stats();
        const char* batch_flag = std::getenv("DS41_MOE_BATCH");
        bool batched = !batch_flag || std::atoi(batch_flag) != 0;
        require(stats.calls == 20, "successful call coverage changed");
        require(stats.batched_calls == (batched ? 20u : 0u), "wrong execution mode");
        require(stats.kernel_launches == (execute ? 3u * f.expected_groups + 20 : batched ? 80u : 510u), "unexpected kernel launches");
        if (peer) {
            require(stats.peer_calls == 20 && stats.peer_fallback_calls == 0, "peer coverage incomplete");
            require(stats.peer_experts == (unsigned)f.expected_remote, "wrong peer routing");
            require(stats.peer_weight_copy_bytes == (execute ? 0 : (uint64_t)f.expected_remote * 3 * f.ff * f.hidden / 32 * 17), "wrong P2P weight bytes");
            require(stats.peer_output_copy_bytes == (execute ? (uint64_t)f.expected_remote * f.hidden * 4 : 0), "wrong P2P output bytes");
            require(stats.peer_activation_copy_bytes == (execute ? (uint64_t)f.expected_remote_calls * f.hidden * 4 : 0), "wrong P2P input bytes");
        }
        const char* pool_flag = std::getenv("DS41_STAGING_POOL");
        const bool pooled = pool_flag && std::atoi(pool_flag) != 0;
        require(stats.staged_slices > 0, "cold staging was not exercised");
        require(pooled ? stats.staging_pool_calls > 0 : stats.staging_pool_calls == 0, "wrong staging mode");
        require(stats.transient_copy_threads == (pooled ? 0 : stats.staged_slices), "unexpected transient workers");
        const char* lut_flag = std::getenv("DS41_MXFP4_REGISTER_LUT");
        const bool register_lut = !lut_flag || std::atoi(lut_flag) != 0;
        require(stats.register_lut_launches == (register_lut ? (execute ? 2u * f.expected_groups : batched ? 40u : 306u) : 0u), "wrong LUT mode");
        const char* coop_flag = std::getenv("DS41_MXFP4_COOPERATIVE");
        const bool cooperative = !coop_flag || std::atoi(coop_flag) != 0;
        require(stats.cooperative_launches == (cooperative ? (execute ? 2u * f.expected_groups : batched ? 40u : 306u) : 0u), "wrong cooperative mode");
        const char* trace_flag = std::getenv("DS41_MOE_TRACE");
        bool traced = trace_flag && std::atoi(trace_flag) != 0;
        require(stats.traced_calls == (traced ? 20u : 0u), "trace coverage incomplete");
        require(stats.trace_failures == (traced ? 1u : 0u), "aborted trace not accounted for");
        for (double t : stats.host_seconds) require(std::isfinite(t) && (traced ? t > 0.0 : t == 0.0), "invalid host trace");
        for (double t : stats.stream_seconds) require(std::isfinite(t) && (traced ? t > 0.0 : t == 0.0), "invalid stream trace");
        if (argc > 2) {
            FILE* file = std::fopen(argv[2], "wb");
            require(file != nullptr, "cannot open output dump");
            size_t wrote = std::fwrite(dump.data(), sizeof(float), dump.size(), file);
            int closed = std::fclose(file);
            require(wrote == dump.size() && closed == 0, "output dump incomplete");
        }
        std::printf("coverage: %llu calls, %llu batched, %llu launches, %llu traced, %llu aborted\n",
            (unsigned long long)stats.calls, (unsigned long long)stats.batched_calls,
            (unsigned long long)stats.kernel_launches, (unsigned long long)stats.traced_calls,
            (unsigned long long)stats.trace_failures);
        std::puts("PASS scoped overlap: bitwise outputs, callback ownership, exception recovery, bounds");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
