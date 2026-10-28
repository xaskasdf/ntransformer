#pragma once

#include "glm5_config.h"
#include <functional>
#include <vector>
#include <string>

namespace nt {
namespace glm5 {

struct TensorRef {
    const uint8_t* data = nullptr;
    DType   dtype = DType::F32;
    int64_t ne[4] = {1, 1, 1, 1};
    std::string name;

    bool valid() const { return data != nullptr; }
    int64_t ne0() const { return ne[0]; }
    int64_t ne1() const { return ne[1]; }
    int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
    int64_t row_bytes() const;
    int64_t block_elems() const;
};

class Model {
public:
    bool load(const std::string& path, std::string& err);

    const Config& cfg() const { return cfg_; }
    const GGUFLoader& loader() const { return loader_; }

    TensorRef tensor(const std::string& name) const;

    void embed(int token, float* out) const;

    void matvec(const TensorRef& w, const float* x, float* y) const;
    void matvec_slice(const TensorRef& w, const float* x, float* y, int64_t row0, int64_t nrows) const;

    void dequant_row(const TensorRef& w, int64_t row, float* out) const;
    float dot_row(const TensorRef& w, int64_t row, const float* x) const;

    int n_threads = 0;

private:
    GGUFLoader loader_;
    Config cfg_;
};

void parallel_for(int64_t n, int n_threads, const std::function<void(int64_t, int)>& fn);

} // namespace glm5
} // namespace nt
