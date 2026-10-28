// Standalone fixed-work benchmark against the actual DS41 Session implementation.
// Link with libntransformer_lib.a; see the accompanying benchmark report.
#include "model/ds41/ds41_forward.h"
#include "model/ds41/ds41_gpu.h"
#include "inference/tokenizer.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ds41;
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32 || c >= 127) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

template<class T> static void array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

static int greedy(const float* logits, int vocab) {
    return int(std::max_element(logits, logits + vocab) - logits);
}

using Profile = std::array<double, 42>;
static const char* names[] = {
    "moe", "attention", "engram", "hc", "other", "qkv", "rope", "compression", "attention_loop", "attention_tail",
    "router", "routed", "shared_cpu", "routed_gpu_calls", "routed_cpu_calls", "shared_gpu_calls", "shared_cpu_calls",
    "routed_upload_bytes", "routed_kernel_launches", "shared_kernel_launches", "tail_kernel_launches",
    "staged_slices", "staging_pool_calls", "transient_copy_threads", "shared_early_calls",
    "register_lut_launches", "prologue_calls", "prologue_readback_waits", "router_gpu_calls",
    "hc_norm", "hc_projection", "hc_mix", "hc_gpu_calls", "hc_cpu_pool_calls", "hc_cpu_serial_calls",
    "resident_attention", "resident_attention_calls", "compressed_upload_bytes", "attention_metadata_upload_bytes",
    "attention_core_launches", "prologue_readback_bytes", "tail_input_upload_bytes"
};
static Profile profile(const Session& s) {
    auto m = gpu_moe_stats();
    const auto& p = s.moe_profile();
    return {s.prof_moe(), s.prof_attn(), s.prof_engram(), s.prof_hc(), s.prof_other(),
        s.prof_qkv(), s.prof_rope(), s.prof_comp(), s.prof_loop(), s.prof_tail(),
        p.router_seconds, p.routed_seconds, p.shared_seconds, double(p.gpu_calls), double(p.cpu_calls),
        double(p.shared_gpu_calls), double(p.shared_cpu_calls), double(m.uploaded_bytes),
        double(m.kernel_launches), double(m.shared_kernel_launches), double(gpu_backbone_stats().tail_gemv_launches),
        double(m.staged_slices), double(m.staging_pool_calls), double(m.transient_copy_threads),
        double(m.shared_early_calls), double(m.register_lut_launches),
        double(gpu_backbone_stats().prologue_calls), double(gpu_backbone_stats().prologue_readback_waits),
        double(gpu_backbone_stats().router_calls), s.hc_profile().norm_seconds,
        s.hc_profile().projection_seconds, s.hc_profile().mix_seconds, double(s.hc_profile().gpu_calls),
        double(s.hc_profile().cpu_pool_calls), double(s.hc_profile().cpu_serial_calls), s.prof_resident_attention(),
        double(gpu_backbone_stats().resident_attention_calls), double(gpu_backbone_stats().resident_comp_upload_bytes),
        double(gpu_backbone_stats().resident_metadata_upload_bytes), double(gpu_backbone_stats().resident_attention_launches),
        double(gpu_backbone_stats().prologue_readback_bytes), double(gpu_backbone_stats().tail_upload_bytes)};
}

static void report(std::ostream& out, const Session& s, int round, const char* phase,
                   int count, double seconds, const Profile& before, const std::vector<double>& latencies) {
    auto after = profile(s);
    out << "{\"kind\":\"phase\",\"round\":" << round << ",\"phase\":" << quoted(phase)
        << ",\"tokens\":" << count << ",\"seconds\":" << seconds << ",\"token_seconds\":";
    array(out, latencies);
    for (size_t i = 0; i < before.size(); ++i) out << "," << quoted(names[i]) << ':' << after[i] - before[i];
    out << "}\n" << std::flush;
    std::fprintf(stderr, "BENCH round=%d phase=%s tokens=%d seconds=%.6f\n", round, phase, count, seconds);
    gpu_print_stats();
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: model corpus prompt_tokens decode_steps warm_repeats output_dir label");
        const auto process_start = Clock::now();
        int n = std::stoi(argv[3]), decode_steps = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        if (n < 2 || n > 4096 || decode_steps < 1 || decode_steps > 256 || repeats < 1 || repeats > 10)
            throw std::runtime_error("invalid benchmark bounds");
        std::filesystem::path output(argv[6]);
        std::filesystem::create_directories(output);
        std::ofstream rows(output / "measurements.jsonl");
        rows << std::setprecision(12);
        Model model; std::string error;
        auto start = Clock::now();
        if (!model.load(argv[1], error)) throw std::runtime_error(error);
        double load_seconds = elapsed(start);
        start = Clock::now();
        nt::Tokenizer tokenizer;
        tokenizer.init(model.loader().vocab(), model.cfg().bos_id, model.cfg().eos_id);
        std::ifstream corpus(argv[2], std::ios::binary);
        if (!corpus) throw std::runtime_error("cannot read corpus");
        std::string text((std::istreambuf_iterator<char>(corpus)), std::istreambuf_iterator<char>());
        auto tokens = tokenizer.encode(text, true);
        if (int(tokens.size()) < n) throw std::runtime_error("corpus too short");
        tokens.resize(n);
        double tokenize_seconds = elapsed(start);
        std::ofstream prompt(output / "prompt.txt", std::ios::binary);
        prompt << tokenizer.decode(tokens); prompt.close();
        std::ofstream ids(output / "prompt-ids.json"); array(ids, tokens); ids << '\n'; ids.close();
        start = Clock::now();
        auto head = model.tensor("output.weight");
        if (head.dtype != nt::DType::BF16 || !gpu_lm_head_init(head.data, head.rows(), head.ne0()))
            throw std::runtime_error("GPU BF16 head unavailable");
        Session session(model);
        if (!session.prof_enabled()) throw std::runtime_error("DS41_PROFILE=1 required");
        double initialize_seconds = elapsed(start);
        rows << "{\"kind\":\"setup\",\"label\":" << quoted(argv[7]) << ",\"load_seconds\":" << load_seconds
             << ",\"tokenize_seconds\":" << tokenize_seconds << ",\"initialize_seconds\":" << initialize_seconds
             << ",\"setup_wall_seconds\":" << elapsed(process_start) << ",\"prompt_tokens\":" << n
             << ",\"decode_steps\":" << decode_steps << ",\"layers\":" << model.cfg().n_layers
             << ",\"vocab\":" << session.vocab_size() << ",\"eos_id\":" << session.eos_id()
             << ",\"shards\":" << model.loader().split_count() << ",\"model_file_bytes\":" << model.loader().total_size()
             << ",\"tokenizer_pre\":" << quoted(model.loader().vocab().pre) << "}\n" << std::flush;
        const size_t width = size_t(session.vocab_size());
        std::vector<float> reference, outputs(width * (decode_steps + 1));
        std::vector<int> reference_ids;
        for (int round = 0; round <= repeats; ++round) {
            session.reset();
            auto before = profile(session);
            std::vector<double> prefill_latencies(n), decode_latencies(decode_steps);
            std::vector<int> predicted;
            predicted.reserve(decode_steps + 1);
            const float* logits = nullptr;
            auto request_start = Clock::now();
            for (int i = 0; i < n; ++i) {
                auto token_start = Clock::now();
                logits = session.forward(tokens[i], i + 1 == n);
                prefill_latencies[i] = elapsed(token_start);
            }
            double prefill_seconds = elapsed(request_start);
            int next = greedy(logits, int(width));
            double ttft = elapsed(request_start);
            predicted.push_back(next);
            std::copy(logits, logits + width, outputs.begin());
            report(rows, session, round, "prefill", n, prefill_seconds, before, prefill_latencies);
            before = profile(session);
            // Fixed work: 64 consumed predictions -> 64 new predictions. EOS is
            // recorded, not a stop condition. Validation and file I/O are outside
            // individual token timings; wall duration also records their overhead.
            auto decode_start = Clock::now();
            for (int i = 0; i < decode_steps; ++i) {
                auto token_start = Clock::now();
                logits = session.forward(next);
                next = greedy(logits, int(width));
                decode_latencies[i] = elapsed(token_start);
                predicted.push_back(next);
                std::copy(logits, logits + width, outputs.begin() + width * (i + 1));
            }
            double decode_wall = elapsed(decode_start), decode_seconds = 0;
            for (double t : decode_latencies) decode_seconds += t;
            report(rows, session, round, "decode", decode_steps, decode_seconds, before, decode_latencies);
            auto after = profile(session);
            if (after[14] - before[14] || after[16] - before[16] ||
                after[13] - before[13] != decode_steps * model.cfg().n_layers ||
                after[15] - before[15] != decode_steps * model.cfg().n_layers)
                throw std::runtime_error("incomplete GPU coverage");
            for (float value : outputs) if (!std::isfinite(value)) throw std::runtime_error("nonfinite logits");
            if (!round) { reference = outputs; reference_ids = predicted; }
            else if (predicted != reference_ids || std::memcmp(reference.data(), outputs.data(), outputs.size() * sizeof(float)))
                throw std::runtime_error("repeat differs from first pass");
            auto eos = std::find(predicted.begin(), predicted.end(), session.eos_id());
            rows << "{\"kind\":\"completion\",\"round\":" << round << ",\"ttft_seconds\":" << ttft
                 << ",\"decode_wall_seconds\":" << decode_wall << ",\"finite_logit_values\":" << outputs.size()
                 << ",\"repeat_bitwise_equal\":true,\"first_eos_prediction\":"
                 << (eos == predicted.end() ? -1 : int(eos - predicted.begin())) << ",\"predicted_ids\":";
            array(rows, predicted); rows << "}\n" << std::flush;
            if (!round) {
                std::ofstream completion(output / "completion.txt", std::ios::binary);
                completion << tokenizer.decode(predicted);
                std::ofstream binary(output / "final-logits.f32", std::ios::binary);
                binary.write(reinterpret_cast<const char*>(logits), width * sizeof(float));
                if (std::getenv("DS41_BENCH_ALL_LOGITS")) {
                    std::ofstream all(output / "all-logits.f32", std::ios::binary);
                    all.write(reinterpret_cast<const char*>(outputs.data()), outputs.size() * sizeof(float));
                    if (!all) throw std::runtime_error("failed to write all logits");
                }
            }
            std::fprintf(stderr, "PASS round=%d finite, repeat exact, EOS index=%d\n", round,
                         eos == predicted.end() ? -1 : int(eos - predicted.begin()));
        }
        if (!rows) throw std::runtime_error("failed to write measurements");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
