#include "model/ds41/ds41_forward.h"
#include "model/ds41/ds41_gpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nt::ds41;
using Clock = std::chrono::steady_clock;
using Phases = std::array<double, 38>;
Phases phases(const Session& s) {
    auto m = gpu_moe_stats();
    return {s.prof_moe(), s.prof_attn(), s.prof_engram(), s.prof_hc(), s.prof_other(),
            s.prof_qkv(), s.prof_rope(), s.prof_comp(), s.prof_loop(), s.prof_tail(),
            s.moe_profile().router_seconds, s.moe_profile().routed_seconds, s.moe_profile().shared_seconds,
            double(s.moe_profile().gpu_calls), double(s.moe_profile().cpu_calls), double(s.moe_profile().overlapped_calls),
            double(gpu_backbone_stats().tail_calls), double(gpu_backbone_stats().grouped_tail_calls), double(gpu_backbone_stats().tail_gemv_launches),
        double(m.calls), double(m.batched_calls), double(m.kernel_launches), double(m.uploaded_bytes), double(m.traced_calls), double(m.trace_failures),
        m.host_seconds[0],m.host_seconds[1],m.host_seconds[2],m.host_seconds[3],m.host_seconds[4],m.host_seconds[5],m.host_seconds[6],m.host_seconds[7],
        m.stream_seconds[0],m.stream_seconds[1],m.stream_seconds[2],m.stream_seconds[3],m.stream_seconds[4]};
}
void report(Session& s, const Phases& before, Clock::time_point start, int round, const char* phase, int tokens) {
    double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    auto after = phases(s);
    const char* names[] = {"moe", "attention", "engram", "hc", "other", "qkv", "rope", "compression", "attention_loop", "attention_tail", "router", "routed", "shared", "gpu_calls", "cpu_calls", "overlapped_calls", "tail_calls", "grouped_tail_calls", "tail_gemv_launches", "moe_calls", "batched_calls", "moe_launches", "uploaded_bytes", "traced_calls", "trace_failures",
        "host_input", "host_lookup", "host_staging", "host_upload_enqueue", "host_launch", "host_callback", "host_wait", "host_publish",
        "stream_input", "stream_prepare_gap", "stream_upload", "stream_compute", "stream_readback"};
    std::printf("{\"round\":%d,\"phase\":\"%s\",\"tokens\":%d,\"seconds\":%.9f", round, phase, tokens, seconds);
    for (int i = 0; i < 38; ++i) std::printf(",\"%s\":%.9f", names[i], after[i] - before[i]);
    std::puts("}"); std::fflush(stdout);
    gpu_print_stats();
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("model tokens repeats output.bin");
        int n = std::stoi(argv[2]), rounds = std::stoi(argv[3]);
        if (n < 1 || n > 256 || rounds < 1 || rounds > 20) throw std::runtime_error("invalid bounds");
        Model model; std::string error;
        if (!model.load(argv[1], error)) throw std::runtime_error(error);
        auto head = model.tensor("output.weight");
        if (head.dtype == nt::DType::BF16 && !gpu_lm_head_init(head.data, head.rows(), head.ne0()))
            throw std::runtime_error("GPU head unavailable");
        Session s(model);
        if (!s.prof_enabled()) throw std::runtime_error("set DS41_PROFILE=1");
        std::vector<float> reference;
        for (int round = -1; round < rounds; ++round) {
            s.reset();
            auto before = phases(s); auto start = Clock::now();
            const float* logits = nullptr;
            for (int i = 0; i < n; ++i) {
                int token = i == 0 ? model.cfg().bos_id : 3 + (i * 37) % (s.vocab_size() - 3);
                logits = s.forward(token, i == n - 1);
            }
            report(s, before, start, round, "prefill", n);
            std::vector<float> outputs(logits, logits + s.vocab_size());
            before = phases(s); start = Clock::now();
            for (int i = 0; i < 4; ++i) {
                int token = (int)(std::max_element(logits, logits + s.vocab_size()) - logits);
                logits = s.forward(token);
                outputs.insert(outputs.end(), logits, logits + s.vocab_size());
            }
            report(s, before, start, round, "decode", 4);
            for (float v : outputs) if (!std::isfinite(v)) throw std::runtime_error("nonfinite logits");
            if (round == -1) reference = outputs;
            else if (std::memcmp(outputs.data(), reference.data(), outputs.size() * sizeof(float)))
                throw std::runtime_error("repeat logits not bitwise equal");
        }
        FILE* f = std::fopen(argv[4], "wb");
        if (!f) throw std::runtime_error("cannot write output");
        size_t wrote = std::fwrite(reference.data(), sizeof(float), reference.size(), f);
        if (std::fclose(f) || wrote != reference.size()) throw std::runtime_error("short output");
        std::puts("PASS warm repeats exact, all logits finite");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
