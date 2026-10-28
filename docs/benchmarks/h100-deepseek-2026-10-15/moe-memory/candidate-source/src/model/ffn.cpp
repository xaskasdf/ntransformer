#include "ffn.h"
#include "../cuda/kernels.h"
#include <cstdio>

namespace nt {

void FFN::init(
    const ModelConfig& config,
    Tensor w_gate, Tensor w_up, Tensor w_down,
    int layer_idx
) {
    hidden_size_       = config.hidden_size;
    intermediate_size_ = config.intermediate_size;
    layer_idx_         = layer_idx;

    gate_dtype_ = w_gate.dtype();
    up_dtype_   = w_up.dtype();
    down_dtype_ = w_down.dtype();

    if (w_gate.device() == Device::CPU) {
        w_gate_ = w_gate.to(Device::CUDA);
        w_up_   = w_up.to(Device::CUDA);
        w_down_ = w_down.to(Device::CUDA);
    } else {
        w_gate_ = std::move(w_gate);
        w_up_   = std::move(w_up);
        w_down_ = std::move(w_down);
    }
}

void FFN::init_streaming(const ModelConfig& config, int layer_idx) {
    hidden_size_       = config.hidden_size;
    intermediate_size_ = config.intermediate_size;
    layer_idx_         = layer_idx;
}

void FFN::set_base_weights(const void* base_gate, const void* base_up,
                           const void* base_down, DType base_dtype) {
    base_gate_ = base_gate;
    base_up_ = base_up;
    base_down_ = base_down;
    base_dtype_ = base_dtype;
}

void FFN::set_delta(const void* u_gate, const void* v_gate,
                    const void* u_up, const void* v_up,
                    const void* u_down, const void* v_down,
                    int rank, float* temp_buf) {
    delta_mode_ = true;
    delta_rank_ = rank;
    delta_u_gate_ = u_gate; delta_v_gate_ = v_gate;
    delta_u_up_ = u_up;     delta_v_up_ = v_up;
    delta_u_down_ = u_down; delta_v_down_ = v_down;
    delta_temp_ = temp_buf;
}

void FFN::set_weights(const void* gate, const void* up, const void* down,
                      DType gate_dt, DType up_dt, DType down_dt) {
    delta_mode_ = false;  // Explicit weights disable delta mode
    gate_dtype_ = gate_dt;
    up_dtype_   = up_dt;
    down_dtype_ = down_dt;

    w_gate_ = Tensor::from_ptr(const_cast<void*>(gate), {intermediate_size_, hidden_size_}, gate_dt, Device::CUDA);
    w_up_   = Tensor::from_ptr(const_cast<void*>(up),   {intermediate_size_, hidden_size_}, up_dt,   Device::CUDA);
    w_down_ = Tensor::from_ptr(const_cast<void*>(down), {hidden_size_, intermediate_size_}, down_dt, Device::CUDA);
}

// Delta GEMV: y = Base*x + U*(V^T*x)
static void delta_gemv(float* y, const void* base_W, DType base_dt,
                       const void* U, const void* V, float* temp_r,
                       const float* x, int out, int in, int rank, void* stream) {
    cuda::launch_gemv(y, base_W, x, out, in, base_dt, stream);             // y = Base*x
    cuda::launch_gemv(temp_r, V, x, rank, in, DType::F16, stream);         // temp = V^T*x
    cuda::launch_gemv_add(y, U, temp_r, out, rank, DType::F16, stream);    // y += U*temp
}

size_t FFN::workspace_size(int seq_len) const {
    // gate_buf: [seq_len * intermediate_size]
    // up_buf:   [seq_len * intermediate_size]
    // silu_buf: [seq_len * intermediate_size]  (reuse gate_buf)
    return 2 * seq_len * intermediate_size_ * sizeof(float);
}

void FFN::forward(
    float* output,
    const float* input,
    int seq_len,
    void* stream
) {
    NT_CHECK(workspace_ != nullptr, "FFN workspace not set");

    float* gate_buf = workspace_;
    float* up_buf = gate_buf + seq_len * intermediate_size_;

    for (int t = 0; t < seq_len; t++) {
        const float* inp = input + t * hidden_size_;
        float* gate_out = gate_buf + t * intermediate_size_;
        float* up_out = up_buf + t * intermediate_size_;
        float* out = output + t * hidden_size_;

        if (delta_mode_) {
            // gate = (Base_gate + U_gate * V_gate^T) * x
            delta_gemv(gate_out, base_gate_, base_dtype_, delta_u_gate_, delta_v_gate_,
                       delta_temp_, inp, intermediate_size_, hidden_size_, delta_rank_, stream);

            // up = (Base_up + U_up * V_up^T) * x
            delta_gemv(up_out, base_up_, base_dtype_, delta_u_up_, delta_v_up_,
                       delta_temp_, inp, intermediate_size_, hidden_size_, delta_rank_, stream);

            // silu_out = SiLU(gate) * up
            cuda::launch_silu_mul(gate_out, gate_out, up_out, intermediate_size_, stream);

            // output = (Base_down + U_down * V_down^T) * silu_out
            delta_gemv(out, base_down_, base_dtype_, delta_u_down_, delta_v_down_,
                       delta_temp_, gate_out, hidden_size_, intermediate_size_, delta_rank_, stream);
        } else {
            // gate = Wgate * x
            cuda::launch_gemv(gate_out, w_gate_.data(), inp,
                intermediate_size_, hidden_size_, gate_dtype_, stream);

            // up = Wup * x
            cuda::launch_gemv(up_out, w_up_.data(), inp,
                intermediate_size_, hidden_size_, up_dtype_, stream);

            // silu_out = SiLU(gate) * up  (in-place into gate_buf)
            cuda::launch_silu_mul(gate_out, gate_out, up_out, intermediate_size_, stream);

            // output = Wdown * silu_out
            cuda::launch_gemv(out, w_down_.data(), gate_out,
                hidden_size_, intermediate_size_, down_dtype_, stream);
        }
    }
}

} // namespace nt
