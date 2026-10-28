#pragma once

// ============================================================
// Cross-platform read-only file mapping + host RAM query
// (POSIX mmap on Linux, CreateFileMapping on Windows)
// ============================================================

#include <cstddef>
#include <cstdint>
#include <string>

namespace nt {

struct FileMap {
    void*   ptr = nullptr;      // mapped view
    uint64_t size = 0;          // file size in bytes
    void*   handle = nullptr;   // Windows mapping object (unused on POSIX)
    bool    valid() const { return ptr != nullptr; }
};

bool map_file_read(const std::string& path, FileMap& out);
void unmap_file(FileMap& m);

// Available physical RAM in bytes (host page-cache aware on Linux).
uint64_t host_avail_ram();
uint64_t host_total_ram();

} // namespace nt

#ifdef _WIN32
#include "filemap_win.inl"
#else
#include "filemap_posix.inl"
#endif
