#include "loader.h"
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <algorithm>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace nt {

GGUFLoader::~GGUFLoader() {
#ifdef _WIN32
    if (mmap_ptr_) {
        UnmapViewOfFile(mmap_ptr_);
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
    }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
    }
#else
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
}

bool GGUFLoader::load(const std::string& path) {
    path_ = path;

#ifdef _WIN32
    // Open file
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Failed to open %s (error %lu)\n", path.c_str(), GetLastError());
        return false;
    }

    // Get file size
    LARGE_INTEGER li;
    if (!GetFileSizeEx(file_handle_, &li)) {
        fprintf(stderr, "Failed to get size of %s\n", path.c_str());
        return false;
    }
    file_size_ = static_cast<size_t>(li.QuadPart);

    // Create file mapping
    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) {
        fprintf(stderr, "Failed to create file mapping for %s (error %lu)\n", path.c_str(), GetLastError());
        return false;
    }

    // Map view of file
    mmap_ptr_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!mmap_ptr_) {
        fprintf(stderr, "Failed to map view of %s (error %lu)\n", path.c_str(), GetLastError());
        return false;
    }
#else
    // Open file
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        fprintf(stderr, "Failed to open %s\n", path.c_str());
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        fprintf(stderr, "Failed to stat %s\n", path.c_str());
        return false;
    }
    file_size_ = st.st_size;

    // mmap the entire file
    mmap_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap %s\n", path.c_str());
        return false;
    }

    // Advise kernel for sequential access
    madvise(mmap_ptr_, file_size_, MADV_SEQUENTIAL);
#endif

    // Parse header
    if (!parse_header()) {
        fprintf(stderr, "Failed to parse GGUF header\n");
        return false;
    }

    return true;
}

bool GGUFLoader::parse_header() {
    const uint8_t* base = static_cast<const uint8_t*>(mmap_ptr_);
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

    fprintf(stderr, "GGUF v%u: %" PRIu64 " tensors, %" PRIu64 " metadata entries\n", version, n_tensors, n_kv);

    // Parse metadata key-value pairs
    std::unordered_map<std::string, std::variant<int, float, std::string, bool>> metadata;
    std::vector<std::string> vocab_tokens;
    std::vector<float> vocab_scores;
    std::vector<int> vocab_types;

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(ptr);
        GGUFType type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
        ptr += 4;

        if (type == GGUFType::ARRAY) {
            // Handle arrays (token list, scores, etc.)
            GGUFType elem_type = static_cast<GGUFType>(*reinterpret_cast<const uint32_t*>(ptr));
            ptr += 4;
            uint64_t n_elems = *reinterpret_cast<const uint64_t*>(ptr);
            ptr += 8;

            if (key == "tokenizer.ggml.tokens" && elem_type == GGUFType::STRING) {
                vocab_tokens.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_tokens.push_back(read_string(ptr));
                }
            } else if (key == "tokenizer.ggml.scores" && elem_type == GGUFType::FLOAT32) {
                vocab_scores.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_scores.push_back(*reinterpret_cast<const float*>(ptr));
                    ptr += 4;
                }
            } else if (key == "tokenizer.ggml.token_type" && elem_type == GGUFType::INT32) {
                vocab_types.reserve(n_elems);
                for (uint64_t j = 0; j < n_elems; j++) {
                    vocab_types.push_back(*reinterpret_cast<const int32_t*>(ptr));
                    ptr += 4;
                }
            } else {
                // Skip array elements we don't need
                for (uint64_t j = 0; j < n_elems; j++) {
                    skip_value(ptr, elem_type);
                }
            }
        } else {
            auto val = read_value(ptr, type);
            metadata[key] = val;
        }
    }

    // Build config
    config_.from_gguf_metadata(metadata);

    // Build vocab
    vocab_.tokens = std::move(vocab_tokens);
    vocab_.scores = std::move(vocab_scores);
    vocab_.token_types = std::move(vocab_types);

    if ((int)vocab_.tokens.size() != config_.vocab_size && !vocab_.tokens.empty()) {
        config_.vocab_size = vocab_.tokens.size();
    }

    // Parse tensor infos
    tensors_.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo& ti = tensors_[i];
        ti.name = read_string(ptr);

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
        DType dt = ggml_to_dtype(ti.ggml_type);
        int64_t n = 1;
        for (auto s : ti.shape) n *= s;
        ti.nbytes = dtype_row_size(dt, n);

        tensor_map_[ti.name] = i;
    }

    // Data section starts at aligned offset after header
    size_t header_size = ptr - base;
    // GGUF aligns data to 32 bytes (or the alignment specified in metadata)
    int alignment = 32;
    auto it = metadata.find("general.alignment");
    if (it != metadata.end()) {
        if (auto* v = std::get_if<int>(&it->second)) {
            alignment = *v;
        }
    }
    data_offset_ = (header_size + alignment - 1) & ~(alignment - 1);
    data_size_ = file_size_ - data_offset_;

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
    return static_cast<const uint8_t*>(mmap_ptr_) + data_offset_ + ti.offset;
}

Tensor GGUFLoader::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    NT_CHECK(it != tensor_map_.end(), ("Tensor not found: " + name).c_str());

    const GGUFTensorInfo& ti = tensors_[it->second];
    DType dt = ggml_to_dtype(ti.ggml_type);

    // Bounds check: verify tensor data is within the mmap'd region
    size_t tensor_end = data_offset_ + ti.offset + ti.nbytes;
    if (tensor_end > file_size_) {
        fprintf(stderr, "\nERROR: Tensor '%s' extends beyond file! "
            "data_offset=%zu, tensor_offset=%zu, nbytes=%zu, file_size=%zu, overrun=%zu\n",
            name.c_str(), data_offset_, (size_t)ti.offset, ti.nbytes, file_size_,
            tensor_end - file_size_);
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
    fprintf(stderr, "File size: %.2f GB\n", file_size_ / (1024.0 * 1024 * 1024));
    fprintf(stderr, "Tensor data: %.2f GB at offset 0x%zX\n",
        data_size_ / (1024.0 * 1024 * 1024), data_offset_);
    fprintf(stderr, "Tensors: %zu\n", tensors_.size());
    fprintf(stderr, "Vocab: %zu tokens\n", vocab_.tokens.size());

    // Print first few tensors
    int count = 0;
    for (const auto& t : tensors_) {
        fprintf(stderr, "  %s: [", t.name.c_str());
        for (int i = 0; i < (int)t.shape.size(); i++) {
            if (i > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%" PRId64, t.shape[i]);
        }
        DType dt = ggml_to_dtype(t.ggml_type);
        fprintf(stderr, "] %s (%.2f MB)\n", dtype_name(dt), t.nbytes / (1024.0 * 1024));
        if (++count >= 20) {
            fprintf(stderr, "  ... and %zu more\n", tensors_.size() - 20);
            break;
        }
    }
}

} // namespace nt
