// Integration regression: terminal logits may be omitted without changing
// causal state. Model files stay external (DS41_TEST_MODEL or first argument).
#include "model/ds41/ds41_forward.h"
#include "model/ds41/ds41_gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void exact(const float* actual, const std::vector<float>& expected) {
    require(actual != nullptr, "missing requested logits");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite logits");
    }
    require(std::memcmp(actual, expected.data(), expected.size() * sizeof(float)) == 0,
            "logits differ bitwise");
}

int greedy(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<float> copy_logits(const float* p, int n) {
    require(p != nullptr, "null logits");
    return {p, p + n};
}

struct Run {
    double seconds;
    int heads;
    std::vector<float> logits;
};

Run prefill(nt::ds41::Session& s, const std::vector<int>& tokens, bool all) {
    s.reset();
    const auto start = std::chrono::steady_clock::now();
    const float* logits = nullptr;
    for (size_t i = 0; i < tokens.size(); ++i) {
        logits = s.forward(tokens[i], all || i + 1 == tokens.size());
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    require(s.head_evaluations() == (all ? (int)tokens.size() : 1), "wrong head count");
    return {seconds, s.head_evaluations(), copy_logits(logits, s.vocab_size())};
}
} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("DS41_TEST_MODEL");
    if (!path || !*path) {
        std::puts("SKIP: set DS41_TEST_MODEL to an external deepseek41 GGUF fixture");
        return 77;
    }
    try {
        const int n = argc > 2 ? std::stoi(argv[2]) : 18;
        const int pairs = argc > 3 ? std::stoi(argv[3]) : 0;
        require(n > 0 && n <= 256 && pairs >= 0 && pairs <= 10, "invalid test bounds");
        nt::ds41::Model model;
        std::string error;
        if (!model.load(path, error)) throw std::runtime_error(error);
        const auto& cfg = model.cfg();
        auto head = model.tensor("output.weight");
        const bool gpu_head = head.dtype == nt::DType::BF16 &&
            nt::ds41::gpu_lm_head_init(head.data, head.rows(), head.ne0());
        std::printf("MODEL layers=%d vocab=%d prompt=%d window=%d topk=%d gpu_head=%d\n",
                    cfg.n_layers, cfg.vocab_size, n, cfg.sliding_window, cfg.indexer_top_k, gpu_head);

        std::vector<int> tokens(n);
        tokens[0] = cfg.bos_id;
        for (int i = 1; i < n; ++i) tokens[i] = 3 + (i * 37) % (cfg.vocab_size - 3);
        nt::ds41::Session reference(model);
        require(reference.logits() == nullptr, "fresh session has stale logits");
        std::vector<std::vector<float>> prompt_logits, continuation_logits;
        for (int token : tokens) {
            // Omitted second argument must keep the original per-token API.
            prompt_logits.push_back(copy_logits(reference.forward(token), cfg.vocab_size));
        }
        require(reference.head_evaluations() == n, "default forward skipped a head");
        std::vector<int> continuation;
        auto next_logits = prompt_logits.back();
        for (int i = 0; i < 4; ++i) {
            const int token = greedy(next_logits);
            continuation.push_back(token);
            next_logits = copy_logits(reference.forward(token), cfg.vocab_size);
            continuation_logits.push_back(next_logits);
        }

        for (bool mixed : {false, true}) {
            nt::ds41::Session candidate(model);
            int heads = 0;
            for (int i = 0; i < n; ++i) {
                const bool compute = i == n - 1 || (mixed && i % 3 == 0);
                const float* logits = candidate.forward(tokens[i], compute);
                if (compute) {
                    exact(logits, prompt_logits[i]);
                    require(logits == candidate.logits(), "logits accessor mismatch");
                    ++heads;
                } else {
                    require(logits == nullptr && candidate.logits() == nullptr, "skipped logits exposed stale data");
                }
                require(candidate.pos() == i + 1, "skipped forward lost position");
                require(candidate.head_evaluations() == heads, "head counter mismatch");
                require(candidate.history_ref() == std::vector<int32_t>(tokens.begin(), tokens.begin() + i + 1),
                        "token history differs");
            }
            for (int i = 0; i < 4; ++i) {
                const auto before = copy_logits(candidate.logits(), cfg.vocab_size);
                require(greedy(before) == continuation[i], "greedy continuation differs");
                exact(candidate.forward(continuation[i]), continuation_logits[i]);
            }
            require(candidate.history_ref() == reference.history_ref(), "continuation history differs");
            candidate.reset();
            require(candidate.logits() == nullptr && candidate.pos() == 0 &&
                    candidate.head_evaluations() == 0 && candidate.history_ref().empty(), "reset incomplete");
            exact(candidate.forward(tokens[0]), prompt_logits[0]);
            require(candidate.head_evaluations() == 1, "single-token prompt skipped its head");
            std::printf("PASS schedule=%s prompt=%d continuation=4 exact=1 reset=1\n", mixed ? "mixed" : "last", n);
        }

        // Optional bounded, in-process benchmark: warm both schedules, then
        // alternate pair order. The same model/cache and token IDs are reused.
        if (pairs > 0) {
            nt::ds41::Session s(model);
            for (bool all : {true, false}) exact(prefill(s, tokens, all).logits.data(), prompt_logits.back());
            for (int pair = 0; pair < pairs; ++pair) {
                for (int item = 0; item < 2; ++item) {
                    const bool all = ((pair + item) % 2 == 0);
                    const Run r = prefill(s, tokens, all);
                    exact(r.logits.data(), prompt_logits.back());
                    std::printf("{\"kind\":\"prefill\",\"pair\":%d,\"all_logits\":%s,\"tokens\":%d,"
                                "\"heads\":%d,\"seconds\":%.9f,\"exact\":true}\n",
                                pair, all ? "true" : "false", n, r.heads, r.seconds);
                    std::fflush(stdout);
                }
            }
        }
        nt::ds41::gpu_print_stats();
        std::puts("PASS DS41 prefill state and logits");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
