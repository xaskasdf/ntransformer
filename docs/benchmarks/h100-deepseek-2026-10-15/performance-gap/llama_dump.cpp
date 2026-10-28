// Same-input comparison with tools/benchmark_ds41.cpp, using the pinned
// llama.cpp runtime/deepseek41 fork's public API. See benchmark RESULTS.md.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static json read_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path.string());
    json value; file >> value; return value;
}
static void floats(const std::filesystem::path& path, const float* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    if (!file) throw std::runtime_error("cannot write " + path.string());
}

static bool dump_active = false;
static bool dump_callback(ggml_tensor* t, bool ask, void*) {
    const char* directory = std::getenv("GAP_DUMP");
    if (!directory || !dump_active || t->type != GGML_TYPE_F32) return false;
    std::string name(t->name);
    static const std::vector<std::string> bases = {"hc_init", "engram_embd", "engram_kv", "engram_out", "hc_attn_pre", "attn_norm", "qr", "q", "kv", "attn_out_raw", "attn_derope", "attn_wo_a", "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_weights", "ffn_moe_out", "ffn_shexp", "ffn_out", "l_last", "hc_out", "result_norm", "result_output"};
    bool wanted=false;
    for (const auto& base:bases) {
        if (name==base || (name.starts_with(base+"-") && name.size()>base.size()+1 && std::isdigit(name[base.size()+1]))) wanted=true;
    }
    if (ask) return wanted;
    if (!wanted) return true;
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    std::vector<float> flat;
    for(int64_t i3=0;i3<t->ne[3];++i3) for(int64_t i2=0;i2<t->ne[2];++i2)
        for(int64_t i1=0;i1<t->ne[1];++i1) for(int64_t i0=0;i0<t->ne[0];++i0) {
            float f; std::memcpy(&f,bytes.data()+i0*t->nb[0]+i1*t->nb[1]+i2*t->nb[2]+i3*t->nb[3],4);flat.push_back(f);
        }
    floats(std::filesystem::path(directory)/(name+".f32"),flat.data(),flat.size());
    std::ofstream manifest(std::filesystem::path(directory)/"tensors.jsonl",std::ios::app);
    json sources=json::array();for(int i=0;i<GGML_MAX_SRC;++i)if(t->src[i])sources.push_back(t->src[i]->name);
    manifest<<json({{"name",name},{"op",ggml_op_name(t->op)},{"buffer",ggml_backend_buffer_name(t->view_src ? t->view_src->buffer : t->buffer)},
      {"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},{"nb",{t->nb[0],t->nb[1],t->nb[2],t->nb[3]}},{"sources",sources}}).dump()<<'\n';
    return true;
}

static int argmax(const float* values, int count) {
    return int(std::max_element(values, values + count) - values);
}
static void fill(llama_batch& batch, const std::vector<int>& ids, int offset, int count,
                 int position, bool logits) {
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = ids[offset + i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits && i + 1 == count;
    }
}
static void decode(llama_context* context, const llama_batch& batch) {
    int result = llama_decode(context, batch);
    if (result) throw std::runtime_error("llama_decode returned " + std::to_string(result));
    llama_synchronize(context);
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("model workload.json output_dir warm_repeats threads cpu_moe ubatch");
        auto process_start = Clock::now();
        auto workload = read_json(argv[2]);
        std::vector<int> prompt = workload.at("prompt_ids").get<std::vector<int>>();
        std::vector<int> continuation = workload.at("continuation_ids").get<std::vector<int>>();
        auto expected = workload.at("expected_prediction_ids").get<std::vector<int>>();
        int repeats = std::stoi(argv[4]), threads = std::stoi(argv[5]);
        int cpu_moe = std::stoi(argv[6]), ubatch = std::stoi(argv[7]);
        if (prompt.empty() || prompt.size() > 512 || continuation.empty() || continuation.size() > 64 ||
            repeats < 1 || repeats > 5 || threads < 1 || threads > 52 || cpu_moe < 31 || cpu_moe > 40 ||
            ubatch < 1 || ubatch > 512 || expected.size() != continuation.size() + 1)
            throw std::runtime_error("invalid bounds");
        std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        std::ofstream rows(output / "measurements.jsonl");
        auto emit = [&](json value) { rows << value.dump() << '\n' << std::flush; };
        // Match normal CLI logging: the library's default callback also prints
        // DEBUG messages (including every CUDA graph replay).
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_INFO) std::fputs(text, stderr);
        }, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) throw std::runtime_error("CUDA backend unavailable");
        ggml_backend_dev_t devices[] = {device, nullptr};
        std::vector<std::string> patterns;
        patterns.reserve(cpu_moe + 1);
        patterns.push_back("\\.engram_embd\\.weight");
        for (int i = 0; i < cpu_moe; ++i)
            patterns.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        std::vector<llama_model_tensor_buft_override> overrides;
        for (const auto& pattern : patterns) overrides.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        overrides.push_back({nullptr, nullptr});
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.lazy_mode = LLAMA_LAZY_MODE_ON;
        mp.tensor_buft_overrides = overrides.data();
        auto start = Clock::now();
        auto* model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        double model_load = seconds(start);
        const auto* vocab = llama_model_get_vocab(model);
        int width = llama_vocab_n_tokens(vocab);
        for (const auto& list : {prompt, continuation})
            for (int id : list) if (id < 0 || id >= width) throw std::runtime_error("invalid token ID");
        auto cp = llama_context_default_params();
        cp.cb_eval = dump_callback;
        cp.n_ctx = 1024;
        cp.n_batch = 512;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        start = Clock::now();
        auto* context = llama_init_from_model(model, cp);
        if (!context) throw std::runtime_error("context init failed");
        double context_init = seconds(start);
        auto batch = llama_batch_init(512, 0, 1);
        emit({{"kind","setup"},{"model_load_seconds",model_load},{"context_init_seconds",context_init},
              {"setup_wall_seconds",seconds(process_start)},{"prompt_tokens",prompt.size()},
              {"decode_steps",continuation.size()},{"vocab",width},{"eos_id",llama_vocab_eos(vocab)},
              {"threads",threads},{"cpu_moe_layers",cpu_moe},{"gpu_moe_layers",40-cpu_moe},
              {"n_batch",512},{"n_ubatch",ubatch},{"n_ctx",1024},{"kv_requested","f32"},
              {"device",ggml_backend_dev_name(device)},{"log_level","info"},{"system_info",llama_print_system_info()}});
        std::vector<float> reference, outputs(size_t(width) * (continuation.size()+1));
        std::vector<int> reference_ids;
        for (int round = 0; round <= repeats; ++round) {
            llama_memory_clear(llama_get_memory(context), true);
            llama_synchronize(context);
            std::vector<int> predictions;
            std::vector<double> chunk_seconds, token_seconds;
            dump_active = round == 0;
            auto request_start = Clock::now();
            for (int offset = 0; offset < int(prompt.size()); offset += ubatch) {
                int count = std::min(ubatch, int(prompt.size()) - offset);
                fill(batch, prompt, offset, count, offset, offset + count == int(prompt.size()));
                auto chunk_start = Clock::now();
                decode(context, batch);
                chunk_seconds.push_back(seconds(chunk_start));
            }
            dump_active = false;
            const float* logits = llama_get_logits_ith(context, -1);
            if (!logits) throw std::runtime_error("missing prefill logits");
            double prefill_seconds = seconds(request_start);
            predictions.push_back(argmax(logits, width));
            double ttft = seconds(request_start);
            std::copy(logits, logits+width, outputs.begin());
            emit({{"kind","phase"},{"round",round},{"phase","prefill"},{"tokens",prompt.size()},
                  {"seconds",prefill_seconds},{"chunk_seconds",chunk_seconds},{"ttft_seconds",ttft}});
            std::fprintf(stderr,"BENCH round=%d prefill=%.6f seconds\n",round,prefill_seconds);
            auto decode_start = Clock::now();
            double decode_seconds = 0;
            for (int i = 0; i < int(continuation.size()); ++i) {
                auto token_start = Clock::now();
                fill(batch, continuation, i, 1, int(prompt.size())+i, true);
                decode(context, batch);
                logits = llama_get_logits_ith(context, -1);
                if (!logits) throw std::runtime_error("missing decode logits");
                predictions.push_back(argmax(logits, width));
                double duration = seconds(token_start);
                token_seconds.push_back(duration); decode_seconds += duration;
                std::copy(logits, logits+width, outputs.begin()+size_t(width)*(i+1));
            }
            emit({{"kind","phase"},{"round",round},{"phase","decode"},{"tokens",continuation.size()},
                  {"seconds",decode_seconds},{"token_seconds",token_seconds},{"wall_seconds",seconds(decode_start)}});
            double max_diff = 0, sum_diff = 0, sum_ref = 0;
            bool bitwise = true;
            for (float value : outputs) if (!std::isfinite(value)) throw std::runtime_error("nonfinite logits");
            if (!round) { reference=outputs; reference_ids=predictions; }
            else {
                bitwise = !std::memcmp(reference.data(),outputs.data(),outputs.size()*sizeof(float));
                for (size_t i=0;i<outputs.size();++i) {
                    double delta=double(outputs[i])-reference[i];
                    max_diff=std::max(max_diff,std::abs(delta));
                    sum_diff+=delta*delta; sum_ref+=double(reference[i])*reference[i];
                }
            }
            int agreement = 0;
            for (size_t i=0;i<predictions.size();++i) agreement += predictions[i]==expected[i];
            emit({{"kind","completion"},{"round",round},{"predicted_ids",predictions},
                  {"agreement_with_ntransformer",agreement},{"finite_logit_values",outputs.size()},
                  {"repeat_bitwise_equal",bitwise},{"repeat_max_abs",max_diff},
                  {"repeat_relative_l2",sum_ref ? std::sqrt(sum_diff/sum_ref) : 0},
                  {"repeat_top1_equal",predictions==reference_ids}});
            if (!round) floats(output/"prompt-logits.f32",outputs.data(),width);
            floats(output/("final-logits-round"+std::to_string(round)+".f32"),logits,width);
            std::fprintf(stderr,"PASS round=%d decode=%.6f seconds; agreement=%d/%zu; replay_bitwise=%d\n",
                         round,decode_seconds,agreement,predictions.size(),int(bitwise));
        }
        if (!rows) throw std::runtime_error("measurement write failed");
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"FAIL %s\n",error.what()); return 1;
    }
}
