#include "norm.h"
#include "../cuda/kernels.h"

namespace nt {

void RMSNorm::init(Tensor weight, float eps) {
    hidden_size_ = weight.size(0);
    eps_ = eps;
    // Transfer weight to GPU if not already there
    if (weight.device() == Device::CPU) {
        weight_ = weight.to(Device::CUDA);
    } else {
        weight_ = std::move(weight);
    }
}

void RMSNorm::forward(float* output, const float* input, int batch_size, void* stream) {
    cuda::launch_rmsnorm(
        output, input,
        weight_.data_as<float>(),
        batch_size, hidden_size_, eps_,
        stream
    );
}

} // namespace nt
