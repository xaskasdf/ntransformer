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

private:
    Tensor w_gate_, w_up_, w_down_;
    DType gate_dtype_, up_dtype_, down_dtype_;

    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int layer_idx_ = 0;

    float* workspace_ = nullptr;
};

} // namespace nt
