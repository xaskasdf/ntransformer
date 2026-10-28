#include "loader.h"
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <algorithm>

namespace nt {

// ============================================================
// GGML type sizes (bytes for `n` elements), independent of DType
// ============================================================
static size_t ggml_type_nbytes(GGMLType t, uint64_t n, bool& known) {
    known = true;
    switch (t) {
        case GGMLType::F32:  return n * 4;
        case GGMLType::F16:  return n * 2;
        case GGMLType::BF16: return n * 2;
        case GGMLType::Q4_0: return (n / 32) * 18;
        case GGMLType::Q4_1: return (n / 32) * 20;
        case GGMLType::Q5_0: return (n / 32) * 22;
        case GGMLType::Q5_1: return (n / 32) * 24;
        case GGMLType::Q8_0: return (n / 32) * 34;
        case GGMLType::Q8_1: return (n / 32) * 40;
        case GGMLType::Q2_K: return (n / 256) * 84;
        case GGMLType::Q3_K: return (n / 256) * 110;
        case GGMLType::Q4_K: return (n / 256) * 144;
        case GGMLType::Q5_K: return (n / 256) * 176;
        case GGMLType::Q6_K: return (n / 256) * 210;
        case GGMLType::Q8_K: return (n / 256) * 292;
        case GGMLType::IQ2_XXS: return (n / 256) * 66;
        case GGMLType::IQ2_XS:  return (n / 256) * 74;
        case GGMLType::IQ3_XXS: return (n / 256) * 98;
        case GGMLType::IQ1_S:   return (n / 256) * 50;
        case GGMLType::IQ4_NL:  return (n / 32) * 18;
        case GGMLType::IQ3_S:   return (n / 256) * 110;
        case GGMLType::IQ2_S:   return (n / 256) * 82;
        case GGMLType::IQ4_XS:  return (n / 256) * 136;
        case GGMLType::I8:  return n;
        case GGMLType::I16: return n * 2;
        case GGMLType::I32: return n * 4;
        case GGMLType::I64: return n * 8;
        case GGMLType::F64: return n * 8;
        case GGMLType::IQ1_M: return (n / 256) * 56;
        case GGMLType::MXFP4: return (n / 32) * 17;
        default:
            known = false;
            return 0;
    }
}

GGUFLoader::~GGUFLoader() {
    for (auto& s : shards_) {
        unmap_file(s.map);
    }
}

bool GGUFLoader::open_shard(Shard& s) {
    if (!map_file_read(s.path, s.map)) {
        fprintf(stderr, "Failed to map %s\n", s.path.c_str());
        return false;
    }
#ifndef _WIN32
    madvise(s.map.ptr, s.map.size, MADV_SEQUENTIAL);
#endif
    return true;
}

bool GGUFLoader::derive_shard_paths(const std::string& path, uint64_t count, std::vector<std::string>& out) const {
    // find the "NNNNN-of-MMMMM" pattern
    auto is_digits = [](const std::string& s, size_t pos, size_t n) {
        if (pos + n > s.size()) return false;
        for (size_t i = 0; i < n; i++) {
            if (s[pos + i] < '0' || s[pos + i] > '9') return false;
        }
        return true;
    };

    size_t of_pos = std::string::npos;
    size_t num_pos = std::string::npos;
    size_t width = 5;
    for (size_t i = 0; i + 5 + 4 <= path.size(); i++) {
        if (path[i] == '-' && is_digits(path, i + 1, 5) &&
            path.compare(i + 6, 4, "-of-") == 0 && is_digits(path, i + 10, 5)) {
            of_pos = i;
            num_pos = i + 1;
            break;
        }
    }
    if (of_pos == std::string::npos) {
        fprintf(stderr, "Cannot derive split shard paths from %s\n", path.c_str());
        return false;
    }

    out.clear();
    for (uint64_t i = 0; i < count; i++) {
        char num[16];
        snprintf(num, sizeof(num), "%0*llu", (int)width, (unsigned long long)(i + 1));
        std::string p = path;
        // skip the first shard (already open)
        p.replace(num_pos, width, num);
        out.push_back(p);
    }
    return true;
}

bool GGUFLoader::load(const std::string& path) {
    path_ = path;

    Shard first;
    first.path = path;
    if (!open_shard(first)) return false;

    // Parse shard 0 fully (stores KV + tensors)
    if (!parse_shard(first, 0, true)) {
        fprintf(stderr, "Failed to parse GGUF header\n");
        unmap_file(first.map);
        return false;
    }

    if (split_no_ != 0) {
        fprintf(stderr, "This is split shard %" PRIu64 "; load the first shard instead\n", split_no_ + 1);
        unmap_file(first.map);
        return false;
    }

    shards_.push_back(first);
    total_size_ += first.map.size;
    data_size_ += first.map.size - first.data_offset;

    // Open remaining shards
    if (split_count_ > 1) {
        std::vector<std::string> paths;
        if (!derive_shard_paths(path, split_count_, paths)) {
            return false;
        }
        for (uint64_t i = 1; i < split_count_; i++) {
            Shard s;
            s.path = paths[i];
            if (!open_shard(s)) return false;
            if (!parse_shard(s, (int)i, false)) {
                fprintf(stderr, "Failed to parse shard %s\n", s.path.c_str());
                return false;
            }
            shards_.push_back(s);
            total_size_ += s.map.size;
            data_size_ += s.map.size - s.data_offset;
        }
        fprintf(stderr, "Loaded %" PRIu64 " split shards, %zu tensors total\n",
                split_count_, tensors_.size());
    }

    // Build model config from metadata
    config_.from_gguf_metadata(config_.metadata);

    if ((int)vocab_.tokens.size() != config_.vocab_size && !vocab_.tokens.empty()) {
        config_.vocab_size = vocab_.tokens.size();
    }

    return true;
}

bool GGUFLoader::parse_shard(Shard& s, int shard_index, bool store) {
    const uint8_t* base = static_cast<const uint8_t*>(s.map.ptr);
    const uint8_t* ptr = base;

    // Magic
    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        return false;
    }

    // Version
    uint32_t version = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return false;
    }

    // Tensor count and metadata KV count
    uint64_t n_tensors = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;
    uint64_t n_kv = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;

    if (store) {
        fprintf(stderr, "GGUF v%u: %" PRIu64 " tensors, %" PRIu64 " metadata entries\n", version, n_tensors, n_kv);
    }

    std::unordered_map<std::string, std::variant<int, float, std::string, bool>> metadata;
    std::vector<std::string> vocab_tokens;
    std::vector<float> vocab_scores;
    std::vector<int> vocab_types;
    std::vector<std::string> vocab_merges;
    std::string vocab_pre;

    auto skip_array = [&](const uint8_t*& p, GGUFType elem_type, uint64_t n_elems) {
        for (uint64_t j = 0; j < n_elems; j++) skip_value(p, elem_type);
    };

    int alignment = alignment_;
    uint64_t split_count_local = 1;
    uint64_t split_no_local = 0;

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(ptr);
        GGUFType type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
        ptr += 4;

        if (type == GGUFType::ARRAY) {
            const uint8_t* arr_ptr = ptr;
            GGUFType elem_type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
            ptr += 4;
            uint64_t n_elems = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;

            if (store && key == "tokenizer.ggml.tokens" && elem_type == GGUFType::STRING) {
                vocab_tokens.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_tokens.push_back(read_string(ptr));
                }
            } else if (store && key == "tokenizer.ggml.scores" && elem_type == GGUFType::FLOAT32) {
                vocab_scores.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_scores.push_back(*reinterpret_cast<const float*>(ptr));
                    ptr += 4;
                }
            } else if (store && key == "tokenizer.ggml.token_type" && elem_type == GGUFType::INT32) {
                vocab_types.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_types.push_back(*reinterpret_cast<const int32_t*>(ptr));
                    ptr += 4;
                }
            } else if (store && key == "tokenizer.ggml.merges" && elem_type == GGUFType::STRING) {
                vocab_merges.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_merges.push_back(read_string(ptr));
                }
            } else if (store) {
                // keep rich arrays (engram constants etc.)
                const uint8_t* rp = arr_ptr;
                GGUFValue v = read_value_rich(rp, type);
                kv_rich_[key] = std::move(v);
                ptr = rp;
            } else {
                skip_array(ptr, elem_type, n_elems);
            }
        } else {
            if (store) {
                GGUFValue v = read_value_rich(ptr, type);
                if (key == "general.alignment" && v.is_numeric()) {
                    alignment = (int)v.i;
                }
                if (key == "split.count") split_count_local = (uint64_t)v.i;
                if (key == "split.no")    split_no_local = (uint64_t)v.i;
                if (key == "tokenizer.ggml.pre" && v.is_string()) {
                    vocab_pre = v.s;
                }
                // legacy metadata view for the Llama ModelConfig path
                {
                    std::variant<int, float, std::string, bool> legacy;
                    switch (type) {
                        case GGUFType::STRING:  legacy = v.s; break;
                        case GGUFType::FLOAT32:
                        case GGUFType::FLOAT64: legacy = (float)v.f; break;
                        case GGUFType::BOOL:    legacy = v.b; break;
                        default:                legacy = (int)v.i; break;
                    }
                    metadata[key] = legacy;
                }
                kv_rich_[key] = std::move(v);
            } else {
                skip_value(ptr, type);
            }
        }
    }

    if (store) {
        split_count_ = split_count_local;
        split_no_ = split_no_local;
        alignment_ = alignment;

        // Build legacy metadata map for ModelConfig (already collected above)
        config_.metadata = metadata;

        // Build vocab
        vocab_.tokens = std::move(vocab_tokens);
        vocab_.scores = std::move(vocab_scores);
        vocab_.token_types = std::move(vocab_types);
        vocab_.merges = std::move(vocab_merges);
        vocab_.pre = std::move(vocab_pre);

        // start data offset calculation right after the KV section (header_size is
        // refined below once tensor infos are parsed)
    }

    // Parse tensor infos
    tensors_.reserve(tensors_.size() + n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo ti;
        ti.name = read_string(ptr);
        ti.shard = shard_index;

        uint32_t n_dims = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += 4;

        ti.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            ti.shape[d] = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;
        }

        ti.ggml_type = static_cast<GGMLType>(*reinterpret_cast<const uint32_t*>(ptr));
        ptr += 4;

        ti.offset = *reinterpret_cast<const uint64_t*>(ptr);
        ptr += 8;

        // Compute nbytes
        uint64_t n = 1;
        for (auto dim : ti.shape) n *= dim;
        bool known = true;
        ti.nbytes = ggml_type_nbytes(ti.ggml_type, n, known);
        if (!known) {
            fprintf(stderr, "Tensor %s uses unsupported GGML type %u\n",
                    ti.name.c_str(), (unsigned)ti.ggml_type);
            return false;
        }

        tensor_map_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }

    // Data section starts at aligned offset after header
    size_t header_size = ptr - base;
    s.data_offset = (header_size + alignment - 1) & ~((size_t)alignment - 1);
    if (store) {
    }

    // Bounds sanity: all tensors must fit inside the shard
    for (const auto& ti : tensors_) {
        if (ti.shard != shard_index) continue;
        size_t end = s.data_offset + (size_t)ti.offset + ti.nbytes;
        if (end > s.map.size) {
            fprintf(stderr, "WARNING: tensor %s extends beyond shard %s (end=%zu, size=%llu)\n",
                    ti.name.c_str(), s.path.c_str(), end, (unsigned long long)s.map.size);
        }
    }

    return true;
}

bool GGUFLoader::parse_header() {
    // Kept for interface compatibility; parse_shard is the real implementation
    return true;
}

std::string GGUFLoader::read_string(const uint8_t*& ptr) const {
    uint64_t len = *reinterpret_cast<const uint64_t*>(ptr);
    ptr += 8;
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

std::variant<int, float, std::string, bool> GGUFLoader::read_value(
    const uint8_t*& ptr, GGUFType type
) const {
    switch (type) {
        case GGUFType::UINT8:   { uint8_t v = *ptr; ptr += 1; return (int)v; }
        case GGUFType::INT8:    { int8_t v = *reinterpret_cast<const int8_t*>(ptr); ptr += 1; return (int)v; }
        case GGUFType::UINT16:  { uint16_t v = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2; return (int)v; }
        case GGUFType::INT16:   { int16_t v = *reinterpret_cast<const int16_t*>(ptr); ptr += 2; return (int)v; }
        case GGUFType::UINT32:  { uint32_t v = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4; return (int)v; }
        case GGUFType::INT32:   { int32_t v = *reinterpret_cast<const int32_t*>(ptr); ptr += 4; return (int)v; }
        case GGUFType::UINT64:  { uint64_t v = *reinterpret_cast<const uint64_t*>(ptr); ptr += 8; return (int)v; }
        case GGUFType::INT64:   { int64_t v = *reinterpret_cast<const int64_t*>(ptr); ptr += 8; return (int)v; }
        case GGUFType::FLOAT32: { float v = *reinterpret_cast<const float*>(ptr); ptr += 4; return v; }
        case GGUFType::FLOAT64: { double v = *reinterpret_cast<const double*>(ptr); ptr += 8; return (float)v; }
        case GGUFType::BOOL:    { bool v = *ptr != 0; ptr += 1; return v; }
        case GGUFType::STRING:  { return read_string(ptr); }
        default: return 0;
    }
}

GGUFValue GGUFLoader::read_value_rich(const uint8_t*& ptr, GGUFType type) const {
    GGUFValue v;
    v.type = type;
    switch (type) {
        case GGUFType::UINT8:   { v.i = *ptr; ptr += 1; break; }
        case GGUFType::INT8:    { v.i = *reinterpret_cast<const int8_t*>(ptr); ptr += 1; break; }
        case GGUFType::UINT16:  { v.i = *reinterpret_cast<const uint16_t*>(ptr); ptr += 2; break; }
        case GGUFType::INT16:   { v.i = *reinterpret_cast<const int16_t*>(ptr); ptr += 2; break; }
        case GGUFType::UINT32:  { v.i = *reinterpret_cast<const uint32_t*>(ptr); ptr += 4; break; }
        case GGUFType::INT32:   { v.i = *reinterpret_cast<const int32_t*>(ptr); ptr += 4; break; }
        case GGUFType::UINT64:  { v.i = (int64_t)*reinterpret_cast<const uint64_t*>(ptr); ptr += 8; break; }
        case GGUFType::INT64:   { v.i = *reinterpret_cast<const int64_t*>(ptr); ptr += 8; break; }
        case GGUFType::FLOAT32: { v.f = *reinterpret_cast<const float*>(ptr); ptr += 4; break; }
        case GGUFType::FLOAT64: { v.f = *reinterpret_cast<const double*>(ptr); ptr += 8; break; }
        case GGUFType::BOOL:    { v.b = *ptr != 0; ptr += 1; break; }
        case GGUFType::STRING:  { v.s = read_string(ptr); break; }
        case GGUFType::ARRAY: {
            GGUFType elem_type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
            ptr += 4;
            uint64_t n_elems = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;
            v.array.elem_type = elem_type;
            switch (elem_type) {
                case GGUFType::UINT8:
                case GGUFType::INT8:
                case GGUFType::UINT16:
                case GGUFType::INT16:
                case GGUFType::UINT32:
                case GGUFType::INT32:
                case GGUFType::UINT64:
                case GGUFType::INT64: {
                    v.array.ints.reserve(n_elems);
                    for (uint64_t j = 0; j < n_elems; j++) {
                        GGUFValue e = read_value_rich(ptr, elem_type);
                        v.array.ints.push_back(e.i);
                    }
                    break;
                }
                case GGUFType::FLOAT32:
                case GGUFType::FLOAT64: {
                    v.array.reals.reserve(n_elems);
                    for (uint64_t j = 0; j < n_elems; j++) {
                        GGUFValue e = read_value_rich(ptr, elem_type);
                        v.array.reals.push_back(e.f);
                    }
                    break;
                }
                case GGUFType::BOOL: {
                    v.array.ints.reserve(n_elems);
                    for (uint64_t j = 0; j < n_elems; j++) {
                        GGUFValue e = read_value_rich(ptr, elem_type);
                        v.array.ints.push_back(e.b ? 1 : 0);
                    }
                    break;
                }
                case GGUFType::STRING: {
                    v.array.strings.reserve(n_elems);
                    for (uint64_t j = 0; j < n_elems; j++) {
                        v.array.strings.push_back(read_string(ptr));
                    }
                    break;
                }
                default:
                    for (uint64_t j = 0; j < n_elems; j++) skip_value(ptr, elem_type);
                    break;
            }
            break;
        }
    }
    return v;
}

void GGUFLoader::skip_value(const uint8_t*& ptr, GGUFType type) const {
    switch (type) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:    ptr += 1; break;
        case GGUFType::UINT16:
        case GGUFType::INT16:   ptr += 2; break;
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32: ptr += 4; break;
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64: ptr += 8; break;
        case GGUFType::STRING:  { read_string(ptr); break; }
        case GGUFType::ARRAY: {
            GGUFType elem_type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
            ptr += 4;
            uint64_t n = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;
            for (uint64_t i = 0; i < n; i++) skip_value(ptr, elem_type);
            break;
        }
    }
}

const GGUFTensorInfo* GGUFLoader::tensor_info(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    return &tensors_[it->second];
}

const void* GGUFLoader::tensor_data(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    const GGUFTensorInfo& ti = tensors_[it->second];
    const Shard& s = shards_[ti.shard];
    return static_cast<const uint8_t*>(s.map.ptr) + s.data_offset + ti.offset;
}

Tensor GGUFLoader::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    NT_CHECK(it != tensor_map_.end(), ("Tensor not found: " + name).c_str());

    const GGUFTensorInfo& ti = tensors_[it->second];
    DType dt = ggml_to_dtype(ti.ggml_type);
    const Shard& s = shards_[ti.shard];

    // Bounds check: verify tensor data is within the mmap'd region
    size_t tensor_end = s.data_offset + ti.offset + ti.nbytes;
    if (tensor_end > s.map.size) {
        fprintf(stderr, "\nERROR: Tensor '%s' extends beyond shard %s! "
            "data_offset=%zu, tensor_offset=%zu, nbytes=%zu, size=%llu, overrun=%zu\n",
            name.c_str(), s.path.c_str(), s.data_offset, (size_t)ti.offset, ti.nbytes,
            (unsigned long long)s.map.size, (size_t)(tensor_end - s.map.size));
        abort();
    }

    void* data = const_cast<void*>(tensor_data(name));
    Tensor t = Tensor::from_ptr(data, ti.shape, dt, Device::CPU);
    t.name = name;
    return t;
}

std::vector<std::string> GGUFLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& t : tensors_) {
        names.push_back(t.name);
    }
    return names;
}

void GGUFLoader::print_info() const {
    fprintf(stderr, "=== GGUF File: %s ===\n", path_.c_str());
    fprintf(stderr, "File size: %.2f GB across %zu shard(s)\n",
        total_size_ / (1024.0 * 1024 * 1024), shards_.size());
    fprintf(stderr, "Tensor data: %.2f GB\n", data_size_ / (1024.0 * 1024 * 1024));
    fprintf(stderr, "Tensors: %zu\n", tensors_.size());
    fprintf(stderr, "Vocab: %zu tokens, %zu merges\n", vocab_.tokens.size(), vocab_.merges.size());

    // Print first few tensors
    int count = 0;
    for (const auto& t : tensors_) {
        fprintf(stderr, "  %s: [", t.name.c_str());
        for (int i = 0; i < (int)t.shape.size(); i++) {
            if (i > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%" PRId64, t.shape[i]);
        }
        DType dt = ggml_to_dtype(t.ggml_type);
        fprintf(stderr, "] %s (%.2f MB, shard %d)\n", dtype_name(dt), t.nbytes / (1024.0 * 1024), t.shard);
        if (++count >= 20) {
            fprintf(stderr, "  ... and %zu more\n", tensors_.size() - 20);
            break;
        }
    }
}

} // namespace nt
