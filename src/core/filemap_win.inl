// Windows implementation (included from filemap.h)
#ifndef NT_FILEMAP_WIN_INL
#define NT_FILEMAP_WIN_INL

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace nt {

inline bool map_file_read(const std::string& path, FileMap& out) {
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER li;
    if (!GetFileSizeEx(file, &li)) { CloseHandle(file); return false; }
    uint64_t size = (uint64_t)li.QuadPart;
    HANDLE mapping = nullptr;
    void* ptr = nullptr;
    if (size > 0) {
        mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) { CloseHandle(file); return false; }
        ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!ptr) { CloseHandle(mapping); CloseHandle(file); return false; }
    }
    CloseHandle(file);   // the mapping object keeps the file alive
    out.ptr = ptr;
    out.size = size;
    out.handle = mapping;
    return true;
}

inline void unmap_file(FileMap& m) {
    if (m.ptr) UnmapViewOfFile(m.ptr);
    if (m.handle) CloseHandle((HANDLE)m.handle);
    m.ptr = nullptr;
    m.handle = nullptr;
    m.size = 0;
}

inline uint64_t host_avail_ram() {
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return (uint64_t)st.ullAvailPhys;
    return 0;
}

inline uint64_t host_total_ram() {
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return (uint64_t)st.ullTotalPhys;
    return 0;
}

} // namespace nt

#endif
