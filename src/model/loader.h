#pragma once

#include "../core/types.h"
#include "../core/tensor.h"
#include "../core/filemap.h"
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
    uint64_t offset;  // offset from this shard's data section start
    int      shard = 0;  // split-GGUF shard index
    size_t   nbytes;  // total bytes
};

// ============================================================
// GGUF vocabulary
// ============================================================
struct GGUFVocab {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int> token_types;  // 0=normal, 1=unknown, 2=control, 3=user, 4=unused, 5=byte
    std::vector<std::string> merges;
    std::string pre;               // tokenizer.ggml.pre
};

// ============================================================
// Rich metadata values (keeps arrays and 64-bit integers)
// ============================================================
struct GGUFArray {
    GGUFType elem_type = GGUFType::UINT32;
    std::vector<int64_t> ints;
    std::vector<double>  reals;
    std::vector<std::string> strings;
};

struct GGUFValue {
    GGUFType type = GGUFType::UINT32;
    int64_t i = 0;
    double  f = 0.0;
    bool    b = false;
    std::string s;
    GGUFArray array;

    bool is_array() const { return type == GGUFType::ARRAY; }
    bool is_string() const { return type == GGUFType::STRING; }
    bool is_numeric() const { return !is_array() && !is_string(); }
};

// ============================================================
// GGUF Model Loader
// Zero-copy via mmap: tensor data stays on disk, loaded on demand
// Supports split GGUFs (00001-of-000NN) transparently.
// ============================================================

class GGUFLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader();

    // Non-copyable
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;

    // Load and parse GGUF file (any shard; KV metadata is taken from shard 0)
    bool load(const std::string& path);

    // Get model config from metadata
    const ModelConfig& config() const { return config_; }
    const GGUFVocab& vocab() const { return vocab_; }

    // Get tensor info by name
    const GGUFTensorInfo* tensor_info(const std::string& name) const;

    // Raw data pointer for a tensor (mmap'd, CPU memory)
    const void* tensor_data(const std::string& name) const;

    // Create a CPU tensor from mmap'd data (zero-copy view)
    Tensor get_tensor(const std::string& name) const;

    // List all tensor names
    std::vector<std::string> tensor_names() const;

    // Total model size (all shards)
    size_t total_size() const { return total_size_; }
    size_t tensor_data_size() const { return data_size_; }

    // Streaming support: raw mmap pointers for async H2D copies (shard 0)
    const void* mmap_data_ptr() const { return static_cast<const uint8_t*>(shards_[0].map.ptr) + shards_[0].data_offset; }
    size_t data_offset() const { return shards_[0].data_offset; }
    const void* mmap_base_ptr() const { return shards_[0].map.ptr; }

    // NVMe direct support: absolute file offsets for LBA calculation
    uint64_t tensor_file_offset(const std::string& name) const {
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) return 0;
        const GGUFTensorInfo& ti = tensors_[it->second];
        return shards_[ti.shard].data_offset + ti.offset;
    }
    int tensor_shard(const std::string& name) const {
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) return -1;
        return tensors_[it->second].shard;
    }
    uint64_t file_data_offset() const { return shards_[0].data_offset; }

    // Split support
    int split_count() const { return (int)shards_.size(); }
    const std::string& shard_path(int i) const { return shards_[i].path; }
    const void* shard_data_base(int i) const {
        return static_cast<const uint8_t*>(shards_[i].map.ptr) + shards_[i].data_offset;
    }
    size_t shard_data_length(int i) const { return shards_[i].map.size - shards_[i].data_offset; }
    uint64_t shard_data_offset(int i) const { return shards_[i].data_offset; }

    // Rich metadata access
    const GGUFValue* kv(const std::string& key) const {
        auto it = kv_rich_.find(key);
        return it == kv_rich_.end() ? nullptr : &it->second;
    }
    int64_t kv_int(const std::string& key, int64_t default_val) const {
        const GGUFValue* v = kv(key);
        if (!v || v->is_array() || v->is_string()) return default_val;
        return (v->type == GGUFType::FLOAT32 || v->type == GGUFType::FLOAT64) ? (int64_t)v->f : v->i;
    }
    float kv_float(const std::string& key, float default_val) const {
        const GGUFValue* v = kv(key);
        if (!v || v->is_array() || v->is_string()) return default_val;
        return (v->type == GGUFType::FLOAT32 || v->type == GGUFType::FLOAT64) ? (float)v->f : (float)v->i;
    }
    std::string kv_string(const std::string& key, const std::string& default_val) const {
        const GGUFValue* v = kv(key);
        if (!v || !v->is_string()) return default_val;
        return v->s;
    }
    bool kv_has(const std::string& key) const { return kv_rich_.count(key) > 0; }
    const std::vector<int64_t>* kv_int_array(const std::string& key) const {
        const GGUFValue* v = kv(key);
        if (!v || !v->is_array()) return nullptr;
        return &v->array.ints;
    }
    const std::vector<double>* kv_float_array(const std::string& key) const {
        const GGUFValue* v = kv(key);
        if (!v || !v->is_array()) return nullptr;
        return &v->array.reals;
    }
    const std::unordered_map<std::string, GGUFValue>& kv_all() const { return kv_rich_; }

    void print_info() const;

private:
    struct Shard {
        std::string path;
        FileMap map;
        size_t data_offset = 0;
    };

    std::string path_;
    std::vector<Shard> shards_;
    size_t total_size_ = 0;
    size_t data_size_ = 0;

    ModelConfig config_;
    GGUFVocab vocab_;

    std::vector<GGUFTensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_map_;  // name -> index in tensors_

    std::unordered_map<std::string, GGUFValue> kv_rich_;
    uint64_t split_count_ = 1;
    uint64_t split_no_ = 0;
    int alignment_ = 32;

    bool open_shard(Shard& s);
    // Parse KV section + tensor infos of one shard.
    // store: keep KV values (only true for shard 0)
    bool parse_shard(Shard& s, int shard_index, bool store);
    bool parse_header();
    std::string read_string(const uint8_t*& ptr) const;
    std::variant<int, float, std::string, bool> read_value(const uint8_t*& ptr, GGUFType type) const;
    GGUFValue read_value_rich(const uint8_t*& ptr, GGUFType type) const;
    void skip_value(const uint8_t*& ptr, GGUFType type) const;

    // Derive shard paths from the pattern "...-00001-of-00010.gguf"
    bool derive_shard_paths(const std::string& path, uint64_t count, std::vector<std::string>& out) const;
};

} // namespace nt
