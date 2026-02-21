#include "transformer.h"
#include "../core/device.h"
#include "../cuda/kernels.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace nt {

// ============================================================
// Destructor
// ============================================================
Transformer::~Transformer() {
    if (norm_weights_gpu_) {
        nt_cuda_free(norm_weights_gpu_);
        norm_weights_gpu_ = nullptr;
    }
}

// ============================================================
// Embedding lookup (CPU -> GPU, unchanged from Phase 1)
// ============================================================

bool Transformer::load(const std::string& gguf_path, int max_context, bool streaming) {
    fprintf(stderr, "Loading model: %s%s\n", gguf_path.c_str(),
        streaming ? " [STREAMING MODE]" : "");

    if (!loader_.load(gguf_path)) {
        return false;
    }

    config_ = loader_.config();

    // Cap context size
    if (config_.max_seq_len > max_context) {
        fprintf(stderr, "Note: Capping context from %d to %d tokens (use --ctx-size to change)\n",
            config_.max_seq_len, max_context);
        config_.max_seq_len = max_context;
    }

    config_.print();
    loader_.print_info();

    // Initialize CUDA device
    if (!CUDADevice::instance().init()) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return false;
    }
    CUDADevice::instance().print_info();

    streaming_mode_ = streaming;

    // Load embedding table (stays on CPU in both modes)
    token_embedding_ = loader_.get_tensor("token_embd.weight");

    // Load output weight -> GPU
    auto* out_info = loader_.tensor_info("output.weight");
    if (out_info) {
        Tensor out_w = loader_.get_tensor("output.weight");
        output_weight_ = out_w.to(Device::CUDA);
    } else {
        Tensor out_w = loader_.get_tensor("token_embd.weight");
        output_weight_ = out_w.to(Device::CUDA);
    }

    // Load output norm
    {
        Tensor norm_w = loader_.get_tensor("output_norm.weight");
        output_norm_.init(std::move(norm_w), config_.norm_eps);
    }

    if (streaming_mode_) {
        load_tiered();
    } else {
        // Resident mode: load all layers to GPU
        layers_.resize(config_.n_layers);
        for (int i = 0; i < config_.n_layers; i++) {
            load_layer(i);
        }
    }

    // Allocate inference buffers (KV cache, workspace, etc.)
    allocate_buffers();

    fprintf(stderr, "Model loaded successfully!%s\n",
        streaming_mode_ ? " (streaming mode)" : "");
    fprintf(stderr, "Free VRAM: %.1f GB\n",
        CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));

    return true;
}

// ============================================================
// Streaming mode: init layers without GPU weights, preload norms
// ============================================================
void Transformer::load_streaming() {
    int n_layers = config_.n_layers;
    layers_.resize(n_layers);

    // Preload all norm weights into a single GPU buffer
    // Each norm is hidden_size floats = hidden_size * 4 bytes
    // Two norms per layer (attn_norm + ffn_norm)
    size_t norm_floats_per_layer = 2 * config_.hidden_size;
    size_t total_norm_floats = (size_t)n_layers * norm_floats_per_layer;
    norm_weights_size_ = total_norm_floats * sizeof(float);

    fprintf(stderr, "Preloading norm weights: %.2f MB for %d layers\n",
        norm_weights_size_ / (1024.0 * 1024.0), n_layers);

    norm_weights_gpu_ = nt_cuda_malloc(norm_weights_size_);
    NT_CHECK(norm_weights_gpu_ != nullptr, "Failed to allocate norm weights GPU buffer");

    float* norm_gpu = static_cast<float*>(norm_weights_gpu_);

    for (int i = 0; i < n_layers; i++) {
        std::string pfx = layer_prefix(i);
        TransformerLayer& layer = layers_[i];

        // Load attn_norm weight -> GPU norm buffer
        {
            Tensor w = loader_.get_tensor(pfx + "attn_norm.weight");
            NT_CHECK(w.dtype() == DType::F32, "Norm weights must be F32");
            float* dst = norm_gpu + i * norm_floats_per_layer;
            nt_cuda_memcpy_h2d(dst, w.data(), config_.hidden_size * sizeof(float));
            layer.attn_norm.init_streaming(config_.hidden_size, config_.norm_eps);
            layer.attn_norm.set_weight(dst);
        }

        // Load ffn_norm weight -> GPU norm buffer
        {
            Tensor w = loader_.get_tensor(pfx + "ffn_norm.weight");
            NT_CHECK(w.dtype() == DType::F32, "Norm weights must be F32");
            float* dst = norm_gpu + i * norm_floats_per_layer + config_.hidden_size;
            nt_cuda_memcpy_h2d(dst, w.data(), config_.hidden_size * sizeof(float));
            layer.ffn_norm.init_streaming(config_.hidden_size, config_.norm_eps);
            layer.ffn_norm.set_weight(dst);
        }

        // Init attention and FFN in streaming mode (no GPU weights yet)
        layer.attention.init_streaming(config_, i);
        layer.ffn.init_streaming(config_, i);

        if ((i + 1) % 10 == 0 || i == n_layers - 1) {
            fprintf(stderr, "  Initialized layer %d/%d (streaming)\n", i + 1, n_layers);
        }
    }

    // Initialize the layer streamer (allocates double buffers)
    streamer_.init(loader_, config_);

    fprintf(stderr, "Streaming setup complete. Buffer size: %.1f MB x 2\n",
        streamer_.buffer_size() / (1024.0 * 1024.0));
}

// ============================================================
// Tiered mode: init layers + 3-tier caching, preload VRAM weights
// ============================================================
void Transformer::load_tiered() {
    int n_layers = config_.n_layers;
    layers_.resize(n_layers);

    // Preload all norm weights into a single GPU buffer (same as load_streaming)
    size_t norm_floats_per_layer = 2 * config_.hidden_size;
    size_t total_norm_floats = (size_t)n_layers * norm_floats_per_layer;
    norm_weights_size_ = total_norm_floats * sizeof(float);

    fprintf(stderr, "Preloading norm weights: %.2f MB for %d layers\n",
        norm_weights_size_ / (1024.0 * 1024.0), n_layers);

    norm_weights_gpu_ = nt_cuda_malloc(norm_weights_size_);
    NT_CHECK(norm_weights_gpu_ != nullptr, "Failed to allocate norm weights GPU buffer");

    float* norm_gpu = static_cast<float*>(norm_weights_gpu_);

    for (int i = 0; i < n_layers; i++) {
        std::string pfx = layer_prefix(i);
        TransformerLayer& layer = layers_[i];

        // Load attn_norm weight -> GPU norm buffer
        {
            Tensor w = loader_.get_tensor(pfx + "attn_norm.weight");
            NT_CHECK(w.dtype() == DType::F32, "Norm weights must be F32");
            float* dst = norm_gpu + i * norm_floats_per_layer;
            nt_cuda_memcpy_h2d(dst, w.data(), config_.hidden_size * sizeof(float));
            layer.attn_norm.init_streaming(config_.hidden_size, config_.norm_eps);
            layer.attn_norm.set_weight(dst);
        }

        // Load ffn_norm weight -> GPU norm buffer
        {
            Tensor w = loader_.get_tensor(pfx + "ffn_norm.weight");
            NT_CHECK(w.dtype() == DType::F32, "Norm weights must be F32");
            float* dst = norm_gpu + i * norm_floats_per_layer + config_.hidden_size;
            nt_cuda_memcpy_h2d(dst, w.data(), config_.hidden_size * sizeof(float));
            layer.ffn_norm.init_streaming(config_.hidden_size, config_.norm_eps);
            layer.ffn_norm.set_weight(dst);
        }

        // Init attention and FFN in streaming mode (no GPU weights yet)
        layer.attention.init_streaming(config_, i);
        layer.ffn.init_streaming(config_, i);

        if ((i + 1) % 10 == 0 || i == n_layers - 1) {
            fprintf(stderr, "  Initialized layer %d/%d (tiered)\n", i + 1, n_layers);
        }
    }

    // Initialize the layer streamer with tiered caching
    streamer_.init_tiered(loader_, config_);

    // For tier A layers: set attention/ffn weights to VRAM-resident pointers permanently
    const auto& tc = streamer_.tier_config();
    for (int i = 0; i < tc.n_vram; i++) {
        LayerWeightPtrs wp = streamer_.get_resident_weights(i);
        layers_[i].attention.set_weights(
            wp.attn_q, wp.attn_k, wp.attn_v, wp.attn_output,
            wp.attn_q_dtype, wp.attn_k_dtype, wp.attn_v_dtype, wp.attn_o_dtype);
        layers_[i].ffn.set_weights(
            wp.ffn_gate, wp.ffn_up, wp.ffn_down,
            wp.ffn_gate_dtype, wp.ffn_up_dtype, wp.ffn_down_dtype);
    }

    fprintf(stderr, "Tiered setup complete. Buffer size: %.1f MB x 2\n",
        streamer_.buffer_size() / (1024.0 * 1024.0));
}

void Transformer::load_layer(int i) {
    std::string pfx = layer_prefix(i);
    TransformerLayer& layer = layers_[i];
    fprintf(stderr, "  Loading layer %d: ", i); fflush(stderr);

    // Attention norm
    fprintf(stderr, "norm "); fflush(stderr);
    {
        Tensor w = loader_.get_tensor(pfx + "attn_norm.weight");
        layer.attn_norm.init(std::move(w), config_.norm_eps);
    }

    // Attention weights
    fprintf(stderr, "attn "); fflush(stderr);
    {
        Tensor wq = loader_.get_tensor(pfx + "attn_q.weight");
        Tensor wk = loader_.get_tensor(pfx + "attn_k.weight");
        Tensor wv = loader_.get_tensor(pfx + "attn_v.weight");
        Tensor wo = loader_.get_tensor(pfx + "attn_output.weight");
        layer.attention.init(config_, std::move(wq), std::move(wk), std::move(wv), std::move(wo), i);
    }

    // FFN norm
    fprintf(stderr, "ffn_norm "); fflush(stderr);
    {
        Tensor w = loader_.get_tensor(pfx + "ffn_norm.weight");
        layer.ffn_norm.init(std::move(w), config_.norm_eps);
    }

    // FFN weights
    fprintf(stderr, "ffn "); fflush(stderr);
    {
        Tensor gate = loader_.get_tensor(pfx + "ffn_gate.weight");
        Tensor up   = loader_.get_tensor(pfx + "ffn_up.weight");
        Tensor down = loader_.get_tensor(pfx + "ffn_down.weight");
        layer.ffn.init(config_, std::move(gate), std::move(up), std::move(down), i);
    }

    fprintf(stderr, "  Loaded layer %d/%d (VRAM free: %.1f GB)\n",
        i + 1, config_.n_layers,
        CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));
    fflush(stderr);
}

void Transformer::allocate_buffers() {
    int max_seq = config_.max_seq_len;
    int hidden = config_.hidden_size;
    int n_kv = config_.n_kv_heads;
    int hd = config_.head_dim;
    int n_layers = config_.n_layers;

    fprintf(stderr, "Allocating buffers (max_seq=%d, hidden=%d, n_kv=%d, hd=%d)...\n",
        max_seq, hidden, n_kv, hd);

    // KV cache: [n_layers, max_seq, n_kv_heads, head_dim] for both K and V
    size_t kv_layer_size = (size_t)max_seq * n_kv * hd;
    fprintf(stderr, "  KV cache: %.1f MB total... ",
        kv_layer_size * sizeof(float) * 2 * n_layers / (1024.0 * 1024));
    fflush(stderr);
    k_cache_ = Tensor::zeros({n_layers, max_seq, n_kv, hd}, DType::F32, Device::CUDA);
    v_cache_ = Tensor::zeros({n_layers, max_seq, n_kv, hd}, DType::F32, Device::CUDA);
    fprintf(stderr, "OK\n");

    // Hidden state buffers
    fprintf(stderr, "  Hidden buffers... ");
    fflush(stderr);
    hidden_buf_ = Tensor::empty({max_seq, hidden}, DType::F32, Device::CUDA);
    residual_buf_ = Tensor::empty({max_seq, hidden}, DType::F32, Device::CUDA);
    fprintf(stderr, "OK\n");

    // Logits buffer
    fprintf(stderr, "  Logits buffer (%d)... ", config_.vocab_size);
    fflush(stderr);
    logits_buf_ = Tensor::empty({config_.vocab_size}, DType::F32, Device::CUDA);
    logits_ = logits_buf_.data_as<float>();
    fprintf(stderr, "OK\n");

    // Workspace for attention + FFN
    size_t attn_ws = (size_t)max_seq * (config_.n_heads + 2 * n_kv + config_.n_heads) * hd;
    size_t ffn_ws = (size_t)2 * max_seq * config_.intermediate_size;
    size_t ws_size = std::max(attn_ws, ffn_ws);
    fprintf(stderr, "  Workspace: %.1f MB (attn=%.1f, ffn=%.1f)... ",
        ws_size * sizeof(float) / (1024.0 * 1024),
        attn_ws * sizeof(float) / (1024.0 * 1024),
        ffn_ws * sizeof(float) / (1024.0 * 1024));
    fflush(stderr);
    workspace_ = Tensor::empty({(int64_t)ws_size}, DType::F32, Device::CUDA);
    fprintf(stderr, "OK\n");

    // Set workspace for all layers
    for (auto& layer : layers_) {
        layer.attention.set_workspace(workspace_.data_as<float>());
        layer.ffn.set_workspace(workspace_.data_as<float>());
    }

    // Positions buffer
    positions_gpu_ = Tensor::empty({max_seq}, DType::I32, Device::CUDA);

    fprintf(stderr, "All buffers allocated. Free VRAM: %.1f GB\n",
        CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));
}

// Convert FP16 (stored as uint16_t) to float, handling subnormals correctly
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // ±zero
        } else {
            // Subnormal FP16 → normalize for FP32
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
    float result;
    memcpy(&result, &f, 4);
    return result;
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
                out[d] = fp16_to_fp32(row[d]);
            }
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else if (emb_dtype == DType::Q8_0) {
        // Q8_0 embedding: 32 weights per block, uint16_t(FP16) scale + 32 int8
        const uint8_t* raw = static_cast<const uint8_t*>(emb_data);
        int blocks_per_row = hidden / 32;
        size_t row_bytes = (size_t)blocks_per_row * sizeof(BlockQ8_0);

        std::vector<float> cpu_buf(seq_len * hidden);
        for (int t = 0; t < seq_len; t++) {
            const uint8_t* row_ptr = raw + (size_t)tokens[t] * row_bytes;
            float* out = cpu_buf.data() + t * hidden;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ8_0* block = reinterpret_cast<const BlockQ8_0*>(row_ptr + b * sizeof(BlockQ8_0));
                float d = fp16_to_fp32(block->d);

                int base = b * 32;
                for (int j = 0; j < 32; j++) {
                    out[base + j] = d * block->qs[j];
                }
            }
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else if (emb_dtype == DType::Q4_0) {
        // Q4_0 embedding: 32 weights per block, uint16_t(FP16) scale + 16 nibble bytes
        const uint8_t* raw = static_cast<const uint8_t*>(emb_data);
        int blocks_per_row = hidden / 32;
        size_t row_bytes = (size_t)blocks_per_row * sizeof(BlockQ4_0);

        std::vector<float> cpu_buf(seq_len * hidden);
        for (int t = 0; t < seq_len; t++) {
            const uint8_t* row_ptr = raw + (size_t)tokens[t] * row_bytes;
            float* out = cpu_buf.data() + t * hidden;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ4_0* block = reinterpret_cast<const BlockQ4_0*>(row_ptr + b * sizeof(BlockQ4_0));
                float d = fp16_to_fp32(block->d);

                int base = b * 32;
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = block->qs[j];
                    int8_t lo = (byte & 0x0F) - 8;
                    int8_t hi = (byte >> 4) - 8;
                    out[base + j]      = d * lo;
                    out[base + j + 16] = d * hi;
                }
            }
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else if (emb_dtype == DType::Q6_K) {
        // Q6_K embedding: 256 weights per block (GGML interleaved layout)
        // ql[128] (lower 4 bits), qh[64] (upper 2 bits), scales[16] (int8), d (FP16)
        // Processed in two 128-weight halves with pointer advancement
        const uint8_t* raw = static_cast<const uint8_t*>(emb_data);
        int blocks_per_row = hidden / 256;
        size_t row_bytes = (size_t)blocks_per_row * sizeof(BlockQ6_K);

        std::vector<float> cpu_buf(seq_len * hidden);
        for (int t = 0; t < seq_len; t++) {
            const uint8_t* row_ptr = raw + (size_t)tokens[t] * row_bytes;
            float* out = cpu_buf.data() + t * hidden;

            for (int b = 0; b < blocks_per_row; b++) {
                const BlockQ6_K* block = reinterpret_cast<const BlockQ6_K*>(row_ptr + b * sizeof(BlockQ6_K));

                float d = fp16_to_fp32(block->d);

                float* y = out + b * 256;
                const uint8_t* ql = block->ql;
                const uint8_t* qh = block->qh;
                const int8_t*  sc = block->scales;

                // Process two 128-weight halves (GGML interleaved layout)
                for (int half = 0; half < 2; half++) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((ql[l + 32]  & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((ql[l]       >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((ql[l + 32]  >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l]      = d * (float)sc[is + 0] * q1;
                        y[l + 32] = d * (float)sc[is + 2] * q2;
                        y[l + 64] = d * (float)sc[is + 4] * q3;
                        y[l + 96] = d * (float)sc[is + 6] * q4;
                    }
                    y += 128; ql += 64; qh += 32; sc += 8;
                }
            }
        }
        nt_cuda_memcpy_h2d(output, cpu_buf.data(), seq_len * hidden * sizeof(float));
    } else {
        fprintf(stderr, "Error: Unsupported embedding dtype: %s\n", dtype_name(emb_dtype));
        nt_cuda_memset(output, 0, seq_len * hidden * sizeof(float));
    }
}

// ============================================================
// Forward pass: dispatch to resident or streaming
// ============================================================
float* Transformer::forward(const int* tokens, int seq_len, int start_pos) {
    if (streaming_mode_) {
        return forward_tiered(tokens, seq_len, start_pos);
    }

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
        layer.attn_norm.forward(residual, hidden_state, seq_len, stream);

        float* k_cache_layer = k_cache_.data_as<float>() + i * kv_layer_stride;
        float* v_cache_layer = v_cache_.data_as<float>() + i * kv_layer_stride;

        layer.attention.forward(
            residual, residual, seq_len, start_pos,
            k_cache_layer, v_cache_layer,
            positions_gpu_.data_as<int>(), stream
        );

        cuda::launch_add_inplace(hidden_state, residual, n, stream);

        // === FFN sub-block with residual connection ===
        layer.ffn_norm.forward(residual, hidden_state, seq_len, stream);

        layer.ffn.forward(residual, residual, seq_len, stream);

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

// ============================================================
// Streaming forward pass: double-buffer pipeline
// ============================================================
float* Transformer::forward_streaming(const int* tokens, int seq_len, int start_pos) {
    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    int hidden = config_.hidden_size;
    int n_layers = config_.n_layers;

    float* hidden_state = hidden_buf_.data_as<float>();
    float* residual = residual_buf_.data_as<float>();

    // 1. Token embedding (same as resident mode)
    embed_tokens(tokens, seq_len, hidden_state, stream);

    // 2. Upload positions to GPU
    std::vector<int> positions(seq_len);
    for (int i = 0; i < seq_len; i++) {
        positions[i] = start_pos + i;
    }
    nt_cuda_memcpy_h2d(positions_gpu_.data(), positions.data(), seq_len * sizeof(int));

    // 3. Pipelined double-buffer through all layers
    //
    // Three-stage pipeline overlapping on different hardware:
    //   Worker thread:  CPU memcpy from mmap to staging[slot]
    //   DMA engine:     async H2D from staging[slot] to gpu[slot]
    //   GPU SMs:        compute layer using gpu[slot]
    //
    // Schedule:
    //   Worker:  [memcpy L0→stg0][memcpy L1→stg1][memcpy L2→stg0]...
    //   H2D:    [               ][stg0→gpu0     ][stg1→gpu1     ]...
    //   Compute:[               ][               ][layer 0       ]...

    int n_kv = config_.n_kv_heads;
    int hd = config_.head_dim;
    int max_seq = config_.max_seq_len;
    size_t kv_layer_stride = max_seq * n_kv * hd;

    // Pre-fill: kick worker to fill staging[0] with layer 0
    streamer_.prefetch_staging(0, 0);

    // Wait for staging ready, then queue H2D for layer 0
    streamer_.begin_h2d(0, 0);

    // Start worker on layer 1 immediately (overlaps with H2D of layer 0)
    if (n_layers > 1) {
        streamer_.prefetch_staging(1, 1);
    }

    for (int i = 0; i < n_layers; i++) {
        int slot = i % 2;
        int next_slot = 1 - slot;

        // Wait for current layer's H2D to complete
        streamer_.wait_transfer(slot);

        // Issue H2D for layer i+1 (staging should be ready or nearly ready)
        if (i + 1 < n_layers) {
            streamer_.begin_h2d(i + 1, next_slot);
        }

        // Kick worker to prefetch layer i+2 into staging[slot]
        // (staging[slot] is now free — its H2D read from staging is done
        //  because we waited on wait_transfer(slot) above, which means
        //  the DMA engine has finished reading from staging[slot])
        if (i + 2 < n_layers) {
            streamer_.prefetch_staging(i + 2, slot);
        }

        // === Compute layer i (overlaps with H2D for i+1 and memcpy for i+2) ===

        LayerWeightPtrs wp = streamer_.get_weights(slot);

        TransformerLayer& layer = layers_[i];
        layer.attention.set_weights(
            wp.attn_q, wp.attn_k, wp.attn_v, wp.attn_output,
            wp.attn_q_dtype, wp.attn_k_dtype, wp.attn_v_dtype, wp.attn_o_dtype);
        layer.ffn.set_weights(
            wp.ffn_gate, wp.ffn_up, wp.ffn_down,
            wp.ffn_gate_dtype, wp.ffn_up_dtype, wp.ffn_down_dtype);

        int n = seq_len * hidden;

        // === Attention sub-block ===
        layer.attn_norm.forward(residual, hidden_state, seq_len, stream);

        float* k_cache_layer = k_cache_.data_as<float>() + i * kv_layer_stride;
        float* v_cache_layer = v_cache_.data_as<float>() + i * kv_layer_stride;

        layer.attention.forward(
            residual, residual, seq_len, start_pos,
            k_cache_layer, v_cache_layer,
            positions_gpu_.data_as<int>(), stream
        );

        cuda::launch_add_inplace(hidden_state, residual, n, stream);

        // === FFN sub-block ===
        layer.ffn_norm.forward(residual, hidden_state, seq_len, stream);
        layer.ffn.forward(residual, residual, seq_len, stream);
        cuda::launch_add_inplace(hidden_state, residual, n, stream);

        // Signal that compute on this slot is done (safe to overwrite GPU buffer)
        streamer_.signal_compute_done(slot);
    }

    // 4. Final norm
    float* last_hidden = hidden_state + (seq_len - 1) * hidden;
    output_norm_.forward(last_hidden, last_hidden, 1, stream);

    // 5. LM head
    cuda::launch_gemv(
        logits_, output_weight_.data(), last_hidden,
        config_.vocab_size, hidden, output_weight_.dtype(), stream
    );

    CUDADevice::instance().synchronize_stream(STREAM_COMPUTE);
    return logits_;
}

// ============================================================
// Tiered forward pass: hybrid VRAM-resident + double-buffer
// ============================================================
float* Transformer::forward_tiered(const int* tokens, int seq_len, int start_pos) {
    void* stream = CUDADevice::instance().stream(STREAM_COMPUTE);
    int hidden = config_.hidden_size;
    int n_layers = config_.n_layers;

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

    // 3. KV cache setup
    int n_kv = config_.n_kv_heads;
    int hd = config_.head_dim;
    int max_seq = config_.max_seq_len;
    size_t kv_layer_stride = max_seq * n_kv * hd;

    const auto& tc = streamer_.tier_config();
    int first_stream = tc.n_vram;  // first non-VRAM layer

    // 4. Kick off pipeline for first streamed layer
    if (first_stream < n_layers) {
        streamer_.prefetch_staging(first_stream, 0);
        streamer_.begin_h2d(first_stream, 0);
        if (first_stream + 1 < n_layers) {
            streamer_.prefetch_staging(first_stream + 1, 1);
        }
    }

    // 5. Process all layers
    for (int i = 0; i < n_layers; i++) {
        TransformerLayer& layer = layers_[i];
        int n = seq_len * hidden;

        if (i < first_stream) {
            // === Tier A: VRAM-resident compute (weights already set in load_tiered) ===

            layer.attn_norm.forward(residual, hidden_state, seq_len, stream);

            float* k_cache_layer = k_cache_.data_as<float>() + i * kv_layer_stride;
            float* v_cache_layer = v_cache_.data_as<float>() + i * kv_layer_stride;

            layer.attention.forward(
                residual, residual, seq_len, start_pos,
                k_cache_layer, v_cache_layer,
                positions_gpu_.data_as<int>(), stream
            );

            cuda::launch_add_inplace(hidden_state, residual, n, stream);

            layer.ffn_norm.forward(residual, hidden_state, seq_len, stream);
            layer.ffn.forward(residual, residual, seq_len, stream);
            cuda::launch_add_inplace(hidden_state, residual, n, stream);
        } else {
            // === Tier B/C: double-buffer streaming pipeline ===
            int stream_idx = i - first_stream;
            int slot = stream_idx % 2;
            int next_slot = 1 - slot;

            // Wait for current layer's H2D to complete
            streamer_.wait_transfer(slot);

            // Issue H2D for next streamed layer
            if (i + 1 < n_layers) {
                streamer_.begin_h2d(i + 1, next_slot);
            }

            // Prefetch layer i+2 into staging[slot]
            if (i + 2 < n_layers) {
                streamer_.prefetch_staging(i + 2, slot);
            }

            // Set weights from double-buffer slot
            LayerWeightPtrs wp = streamer_.get_weights(slot);
            layer.attention.set_weights(
                wp.attn_q, wp.attn_k, wp.attn_v, wp.attn_output,
                wp.attn_q_dtype, wp.attn_k_dtype, wp.attn_v_dtype, wp.attn_o_dtype);
            layer.ffn.set_weights(
                wp.ffn_gate, wp.ffn_up, wp.ffn_down,
                wp.ffn_gate_dtype, wp.ffn_up_dtype, wp.ffn_down_dtype);

            // Compute
            layer.attn_norm.forward(residual, hidden_state, seq_len, stream);

            float* k_cache_layer = k_cache_.data_as<float>() + i * kv_layer_stride;
            float* v_cache_layer = v_cache_.data_as<float>() + i * kv_layer_stride;

            layer.attention.forward(
                residual, residual, seq_len, start_pos,
                k_cache_layer, v_cache_layer,
                positions_gpu_.data_as<int>(), stream
            );

            cuda::launch_add_inplace(hidden_state, residual, n, stream);

            layer.ffn_norm.forward(residual, hidden_state, seq_len, stream);
            layer.ffn.forward(residual, residual, seq_len, stream);
            cuda::launch_add_inplace(hidden_state, residual, n, stream);

            // Signal compute done (safe to overwrite GPU buffer)
            streamer_.signal_compute_done(slot);
        }
    }

    // 6. Final norm
    float* last_hidden = hidden_state + (seq_len - 1) * hidden;
    output_norm_.forward(last_hidden, last_hidden, 1, stream);

    // 7. LM head
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
