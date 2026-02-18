#pragma once

#include "../core/types.h"
#include "../core/tensor.h"
#include "config.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace nt {

// ============================================================
// GGUF tensor info (parsed from header, before mmap)
// ============================================================
struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    GGMLType ggml_type;
    uint64_t offset;  // offset from data section start
    size_t   nbytes;  // total bytes
};

// ============================================================
// GGUF vocabulary
// ============================================================
struct GGUFVocab {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int> token_types;  // 0=normal, 1=unknown, 2=control, 3=user, 4=unused, 5=byte
};

// ============================================================
// GGUF Model Loader
// Zero-copy via mmap: tensor data stays on disk, loaded on demand
// ============================================================

class GGUFLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader();

    // Non-copyable
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    // Load and parse GGUF file
    bool load(const std::string& path);

    // Get model config from metadata
    const ModelConfig& config() const { return config_; }
    const GGUFVocab& vocab() const { return vocab_; }

    // Get tensor info by name
    const GGUFTensorInfo* tensor_info(const std::string& name) const;

    // Get raw data pointer for a tensor (mmap'd, CPU memory)
    const void* tensor_data(const std::string& name) const;

    // Create a CPU tensor from mmap'd data (zero-copy view)
    Tensor get_tensor(const std::string& name) const;

    // List all tensor names
    std::vector<std::string> tensor_names() const;

    // Total model size
    size_t total_size() const { return file_size_; }
    size_t tensor_data_size() const { return data_size_; }

    void print_info() const;

private:
    std::string path_;
    int fd_ = -1;
    void* mmap_ptr_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;  // byte offset where tensor data begins
    size_t data_size_ = 0;

    ModelConfig config_;
    GGUFVocab vocab_;

    std::vector<GGUFTensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_map_;  // name -> index in tensors_

    // Parsing helpers
    bool parse_header();
    std::string read_string(const uint8_t*& ptr) const;
    std::variant<int, float, std::string, bool> read_value(const uint8_t*& ptr, GGUFType type) const;
    void skip_value(const uint8_t*& ptr, GGUFType type) const;
};

} // namespace nt
