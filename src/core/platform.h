#pragma once

// Platform compatibility: fill POSIX gaps for Windows builds.
// Only included where needed; the Linux path is unchanged.

#ifdef _WIN32

// ---- aligned alloc ----
#include <malloc.h>
inline void* nt_win32_aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
inline void nt_win32_aligned_free(void* ptr) {
    _aligned_free(ptr);
}
#define NT_ALIGNED_ALLOC(align, size) nt_win32_aligned_alloc(align, size)
#define NT_ALIGNED_FREE(ptr)          nt_win32_aligned_free(ptr)

// ---- mmap shim (read-only, private) ----
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

struct NtMmap {
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE map_handle  = NULL;
    void*  ptr         = nullptr;
    size_t size        = 0;
};

inline NtMmap nt_mmap_open(const char* path) {
    NtMmap m;
    m.file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m.file_handle == INVALID_HANDLE_VALUE) return m;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(m.file_handle, &sz)) {
        CloseHandle(m.file_handle);
        m.file_handle = INVALID_HANDLE_VALUE;
        return m;
    }
    m.size = (size_t)sz.QuadPart;

    m.map_handle = CreateFileMappingA(m.file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m.map_handle) {
        CloseHandle(m.file_handle);
        m.file_handle = INVALID_HANDLE_VALUE;
        return m;
    }

    m.ptr = MapViewOfFile(m.map_handle, FILE_MAP_READ, 0, 0, 0);
    if (!m.ptr) {
        CloseHandle(m.map_handle);
        CloseHandle(m.file_handle);
        m.map_handle  = NULL;
        m.file_handle = INVALID_HANDLE_VALUE;
    }
    return m;
}

inline void nt_mmap_close(NtMmap& m) {
    if (m.ptr)         { UnmapViewOfFile(m.ptr);  m.ptr = nullptr; }
    if (m.map_handle)  { CloseHandle(m.map_handle);  m.map_handle = NULL; }
    if (m.file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m.file_handle);
        m.file_handle = INVALID_HANDLE_VALUE;
    }
}

// ---- available RAM (bytes) ----
#include <psapi.h>
inline size_t nt_available_ram() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t)ms.ullAvailPhys;
}
inline size_t nt_total_ram() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t)ms.ullTotalPhys;
}

#else // POSIX

#define NT_ALIGNED_ALLOC(align, size) aligned_alloc(align, size)
#define NT_ALIGNED_FREE(ptr)          ::free(ptr)

#endif // _WIN32
