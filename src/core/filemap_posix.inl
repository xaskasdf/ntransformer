// POSIX implementation (included from filemap.h)
#ifndef NT_FILEMAP_POSIX_INL
#define NT_FILEMAP_POSIX_INL

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>

namespace nt {

inline bool map_file_read(const std::string& path, FileMap& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    uint64_t size = (uint64_t)st.st_size;
    void* ptr = nullptr;
    if (size > 0) {
        ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); return false; }
    }
    close(fd);
    out.ptr = ptr;
    out.size = size;
    out.handle = nullptr;
    return true;
}

inline void unmap_file(FileMap& m) {
    if (m.ptr) munmap(m.ptr, m.size);
    m.ptr = nullptr;
    m.handle = nullptr;
    m.size = 0;
}

inline uint64_t host_avail_ram() {
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (uint64_t)si.freeram * si.mem_unit + (uint64_t)si.bufferram * si.mem_unit;
    }
    return 0;
}

inline uint64_t host_total_ram() {
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        return (uint64_t)si.totalram * si.mem_unit;
    }
    return 0;
}

} // namespace nt

#endif
