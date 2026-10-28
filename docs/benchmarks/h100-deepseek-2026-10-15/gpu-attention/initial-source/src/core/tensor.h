#pragma once

#include "types.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace nt {

class Allocator; // forward decl

// ============================================================
// Tensor: multi-device tensor with view support
// ============================================================
class Tensor {
public:
    Tensor() = default;
    ~Tensor();

    // No copy, move only
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Factory methods
    static Tensor empty(const std::vector<int64_t>& shape, DType dtype, Device device);
    static Tensor from_ptr(void* data, const std::vector<int64_t>& shape, DType dtype, Device device);
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype, Device device);

    // View operations (no data copy)
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;

    // Data access
    void* data() { return data_; }
    const void* data() const { return data_; }
    template<typename T> T* data_as() { return static_cast<T*>(data_); }
    template<typename T> const T* data_as() const { return static_cast<const T*>(data_); }

    // Metadata
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int64_t size(int dim) const { return shape_[dim]; }
    int64_t numel() const;
    size_t nbytes() const;
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool is_contiguous() const;
    bool is_view() const { return is_view_; }

    // Device transfer
    Tensor to(Device target) const;
    void copy_from(const Tensor& src);

    // Debug
    std::string to_string() const;
    void print_info() const;

    // Name for debugging / model loading
    std::string name;

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DType dtype_ = DType::F32;
    Device device_ = Device::CPU;
    bool is_view_ = false;
    bool owns_data_ = false;
    size_t offset_ = 0;  // byte offset for views

    // Shared ownership for views
    std::shared_ptr<void> data_owner_;

    void compute_strides();
    void allocate();
    void free();
};

} // namespace nt
