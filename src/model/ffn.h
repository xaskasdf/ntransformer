#pragma once

#include "../core/tensor.h"
#include "config.h"

namespace nt {

// ============================================================
// SwiGLU Feed-Forward Network
// output = Wo * (SiLU(Wgate * x) * Wup * x)
//
// Wgate: [intermediate_size, hidden_size]
// Wup:   [intermediate_size, hidden_size]
// Wdown: [hidden_size, intermediate_size]
// ============================================================

class FFN {
public:
    void init(
        const ModelConfig& config,
        Tensor w_gate, Tensor w_up, Tensor w_down,
        int layer_idx
    );

    // Streaming mode: store config only, skip GPU weight transfer
    void init_streaming(const ModelConfig& config, int layer_idx);

    // Set weights to point at GPU buffer (non-owning views)
    void set_weights(const void* gate, const void* up, const void* down,
                     DType gate_dt, DType up_dt, DType down_dt);

    void forward(
        float* output,
        const float* input,
        int seq_len,
        void* stream
    );

    size_t workspace_size(int seq_len) const;
    void set_workspace(float* ptr) { workspace_ = ptr; }

    // Delta encoding: set permanent base weights (VRAM, called once at init)
    void set_base_weights(const void* base_gate, const void* base_up,
                          const void* base_down, DType base_dtype);

    // Delta encoding: set per-layer delta (called per streaming iteration)
    void set_delta(const void* u_gate, const void* v_gate,
                   const void* u_up, const void* v_up,
                   const void* u_down, const void* v_down,
                   int rank, float* temp_buf);

    bool is_delta_mode() const { return delta_mode_; }

private:
    Tensor w_gate_, w_up_, w_down_;
    DType gate_dtype_, up_dtype_, down_dtype_;

    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int layer_idx_ = 0;

    float* workspace_ = nullptr;

    // Delta encoding state
    bool delta_mode_ = false;
    int delta_rank_ = 0;
    const void* base_gate_ = nullptr;
    const void* base_up_ = nullptr;
    const void* base_down_ = nullptr;
    DType base_dtype_ = DType::Q6_K;
    const void* delta_u_gate_ = nullptr; const void* delta_v_gate_ = nullptr;
    const void* delta_u_up_ = nullptr;   const void* delta_v_up_ = nullptr;
    const void* delta_u_down_ = nullptr; const void* delta_v_down_ = nullptr;
    float* delta_temp_ = nullptr;
};

} // namespace nt
