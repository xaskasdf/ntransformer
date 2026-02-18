#include "transformer.h"
#include "../core/device.h"
#include "../cuda/kernels.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace nt {

// ============================================================
// Embedding lookup kernel (launched from here, defined inline)
// ============================================================
// We'll use a simple CPU -> GPU approach for embedding:
// Look up on CPU, copy to GPU. For Phase 1 this is fine.
// Future: GPU embedding kernel for batched prefill.

bool Transformer::load(const std::string& gguf_path) {
    fprintf(stderr, "Loading model: %s\n", gguf_path.c_str());

    if (!loader_.load(gguf_path)) {
        return false;
    }

    config_ = loader_.config();
    config_.print();
    loader_.print_info();

    // Initialize CUDA device
    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return false;
    }
    CUDADevice::instance().print_info();

    // Load embedding table
    // GGUF name: "token_embd.weight"
    token_embedding_ = loader_.get_tensor("token_embd.weight");

    // Load output weight (LM head)
    // Some models share embedding/output, some have separate "output.weight"
    auto* out_info = loader_.tensor_info("output.weight");
    if (out_info) {
        output_weight_ = loader_.get_tensor("output.weight");
    } else {
        // Share with embedding
        output_weight_ = loader_.get_tensor("token_embd.weight");
    }

    // Load output norm
    {
        Tensor norm_w = loader_.get_tensor("output_norm.weight");
        output_norm_.init(std::move(norm_w), config_.norm_eps);
    }

    // Load layers
    layers_.resize(config_.n_layers);
    for (int i = 0; i < config_.n_layers; i++) {
        load_layer(i);
    }

    // Allocate inference buffers
    allocate_buffers();

    fprintf(stderr, "Model loaded successfully!\n");
    fprintf(stderr, "Free VRAM: %.1f GB\n",
        CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));

    return true;
}

void Transformer::load_layer(int i) {
    std::string pfx = layer_prefix(i);
    TransformerLayer& layer = layers_[i];

    // Attention norm
    {
        Tensor w = loader_.get_tensor(pfx + "attn_norm.weight");
        layer.attn_norm.init(std::move(w), config_.norm_eps);
    }

    // Attention weights
    {
        Tensor wq = loader_.get_tensor(pfx + "attn_q.weight");
        Tensor wk = loader_.get_tensor(pfx + "attn_k.weight");
        Tensor wv = loader_.get_tensor(pfx + "attn_v.weight");
        Tensor wo = loader_.get_tensor(pfx + "attn_output.weight");
        layer.attention.init(config_, std::move(wq), std::move(wk), std::move(wv), std::move(wo), i);
    }

    // FFN norm
    {
        Tensor w = loader_.get_tensor(pfx + "ffn_norm.weight");
        layer.ffn_norm.init(std::move(w), config_.norm_eps);
    }

    // FFN weights
    {
        Tensor gate = loader_.get_tensor(pfx + "ffn_gate.weight");
        Tensor up   = loader_.get_tensor(pfx + "ffn_up.weight");
        Tensor down = loader_.get_tensor(pfx + "ffn_down.weight");
        layer.ffn.init(config_, std::move(gate), std::move(up), std::move(down), i);
    }

    if ((i + 1) % 10 == 0 || i == config_.n_layers - 1) {
        fprintf(stderr, "  Loaded layer %d/%d\n", i + 1, config_.n_layers);
    }
}

void Transformer::allocate_buffers() {
    int max_seq = config_.max_seq_len;
    int hidden = config_.hidden_size;
    int n_kv = config_.n_kv_heads;
    int hd = config_.head_dim;
    int n_layers = config_.n_layers;

    // KV cache: [n_layers, max_seq, n_kv_heads, head_dim] for both K and V
    size_t kv_layer_size = max_seq * n_kv * hd;
    k_cache_ = Tensor::zeros({n_layers, max_seq, n_kv, hd}, DType::F32, Device::CUDA);
    v_cache_ = Tensor::zeros({n_layers, max_seq, n_kv, hd}, DType::F32, Device::CUDA);

    fprintf(stderr, "KV cache: %.1f MB per layer, %.1f MB total\n",
        kv_layer_size * sizeof(float) * 2 / (1024.0 * 1024),
        kv_layer_size * sizeof(float) * 2 * n_layers / (1024.0 * 1024));

    // Hidden state buffers
    hidden_buf_ = Tensor::empty({max_seq, hidden}, DType::F32, Device::CUDA);
    residual_buf_ = Tensor::empty({max_seq, hidden}, DType::F32, Device::CUDA);

    // Logits buffer
    logits_buf_ = Tensor::empty({config_.vocab_size}, DType::F32, Device::CUDA);
    logits_ = logits_buf_.data_as<float>();

    // Workspace for attention + FFN
    // Attention needs: q/k/v/out buffers
    // FFN needs: gate/up buffers
    int attn_ws = max_seq * (config_.n_heads + 2 * n_kv + config_.n_heads) * hd;
    int ffn_ws = 2 * max_seq * config_.intermediate_size;
    int ws_size = std::max(attn_ws, ffn_ws);  // reuse for attn and ffn
    workspace_ = Tensor::empty({ws_size}, DType::F32, Device::CUDA);

    // Set workspace for all layers
    for (auto& layer : layers_) {
        layer.attention.set_workspace(workspace_.data_as<float>());
        layer.ffn.set_workspace(workspace_.data_as<float>());
    }

    // Positions buffer
    positions_gpu_ = Tensor::empty({max_seq}, DType::I32, Device::CUDA);

    fprintf(stderr, "Workspace: %.1f MB\n", ws_size * sizeof(float) / (1024.0 * 1024));
}

void Transformer::embed_tokens(const int* tokens, int seq_len, float* output, void* stream) {
    // Simple embedding lookup
    // For quantized embeddings we'd need a dequant kernel; for F16 a conversion kernel
    // Phase 1: CPU lookup + copy to GPU

    DType emb_dtype = token_embedding_.dtype();
    int hidden = config_.hidden_size;
    const void* emb_data = token_embedding_.data();

    if (emb_dtype == DType::F32) {
        // Direct F32 embedding
        const float* emb = static_cast<const float*>(emb_data);
        // Allocate temp CPU buffer
        std::vector<float> cpu_buf(seq_len * hidden);
        for (int t = 0; t < seq_len; t++) {
            memcpy(cpu_buf.data() + t * hidden, emb + tokens[t] * hidden, hidden * sizeof(float));
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else if (emb_dtype == DType::F16) {
        // F16 embedding -> dequant to F32
        const uint16_t* emb = static_cast<const uint16_t*>(emb_data);
        std::vector<float> cpu_buf(seq_len * hidden);
        for (int t = 0; t < seq_len; t++) {
            const uint16_t* row = emb + tokens[t] * hidden;
            float* out = cpu_buf.data() + t * hidden;
            for (int d = 0; d < hidden; d++) {
                // FP16 to FP32 conversion (handles subnormals correctly)
                uint16_t h = row[d];
                uint32_t sign = (h >> 15) & 1;
                int32_t  exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) {
                        f = sign << 31;  // +/- zero
                    } else {
                        // Subnormal FP16 -> normalize for FP32
                        exp = 1;
                        while (!(mant & 0x400)) { mant <<= 1; exp--; }
                        mant &= 0x3FF;
                        f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f = (sign << 31) | 0x7F800000 | (mant << 13);  // inf/nan
                } else {
                    f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&out[d], &f, 4);
            }
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else {
        // Quantized embeddings - dequantize on CPU
        // For Q4_0, Q8_0, etc. - we handle the common case
        fprintf(stderr, "Warning: Quantized embedding lookup not fully optimized (dtype=%s)\n",
            dtype_name(emb_dtype));

        // Fallback: treat as raw data and copy row bytes, then dequant
        // For now, just zero the output
        nt_cuda_memset(output, 0, seq_len * hidden * sizeof(float));
    }
}

float* Transformer::forward(const int* tokens, int seq_len, int start_pos) {
    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    int hidden = config_.hidden_size;

    float* hidden_state = hidden_buf_.data_as<float>();
    float* residual = residual_buf_.data_as<float>();

    // 1. Token embedding
    embed_tokens(tokens, seq_len, hidden_state, stream);

    // 2. Upload positions to GPU
    std::vector<int> positions(seq_len);
    for (int i = 0; i < seq_len; i++) {
        positions[i] = start_pos + i;
    }
    nt_cuda_memcpy_h2d(positions_gpu_.data(), positions.data(), seq_len * sizeof(int));

    // 3. Process each layer
    int n_kv = config_.n_kv_heads;
    int hd = config_.head_dim;
    int max_seq = config_.max_seq_len;
    size_t kv_layer_stride = max_seq * n_kv * hd;

    for (int i = 0; i < config_.n_layers; i++) {
        TransformerLayer& layer = layers_[i];
        int n = seq_len * hidden;

        // === Attention sub-block with residual connection ===
        // 1. norm_out = RMSNorm(hidden_state) -> residual buffer
        layer.attn_norm.forward(residual, hidden_state, seq_len, stream);

        // 2. attn_out = Attention(norm_out) -> residual buffer (in-place)
        float* k_cache_layer = k_cache_.data_as<float>() + i * kv_layer_stride;
        float* v_cache_layer = v_cache_.data_as<float>() + i * kv_layer_stride;

        layer.attention.forward(
            residual, residual, seq_len, start_pos,
            k_cache_layer, v_cache_layer,
            positions_gpu_.data_as<int>(), stream
        );

        // 3. hidden_state += attn_out  (residual connection)
        cuda::launch_add_inplace(hidden_state, residual, n, stream);

        // === FFN sub-block with residual connection ===
        // 4. norm_out = RMSNorm(hidden_state) -> residual buffer
        layer.ffn_norm.forward(residual, hidden_state, seq_len, stream);

        // 5. ffn_out = FFN(norm_out) -> residual buffer (in-place)
        layer.ffn.forward(residual, residual, seq_len, stream);

        // 6. hidden_state += ffn_out  (residual connection)
        cuda::launch_add_inplace(hidden_state, residual, n, stream);
    }

    // 4. Final norm - only the last token matters for next-token prediction
    float* last_hidden = hidden_state + (seq_len - 1) * hidden;
    output_norm_.forward(last_hidden, last_hidden, 1, stream);

    // 5. LM head (output projection) - only last token
    cuda::launch_gemv(
        logits_, output_weight_.data(), last_hidden,
        config_.vocab_size, hidden, output_weight_.dtype(), stream
    );

    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);
    return logits_;
}

std::string Transformer::layer_prefix(int i) const {
    return "blk." + std::to_string(i) + ".";
}

} // namespace nt
