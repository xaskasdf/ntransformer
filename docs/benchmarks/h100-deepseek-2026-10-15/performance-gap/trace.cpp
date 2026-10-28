#include "trace.h"
#include <cupti.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cxxabi.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static FILE* output;
static std::mutex mutex;
static bool initialized;
static uint64_t start;
static void crash(int) { void* frames[40]; int n=backtrace(frames,40);backtrace_symbols_fd(frames,n,2);_exit(139); }
static void check(CUptiResult r) {
    if (r != CUPTI_SUCCESS) { const char* s; cuptiGetResultString(r, &s); throw std::runtime_error(s); }
}
static void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8 * 1024 * 1024; *buffer = (uint8_t*)std::malloc(*size); *max_records = 0;
}
static void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!output) { std::free(buffer); return; }
    CUpti_Activity* record;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = (CUpti_ActivityKernel10*)record;
            int status; char* demangled = abi::__cxa_demangle(k->name, nullptr, nullptr, &status);
            std::fprintf(output, "kernel\t%s\t%llu\t%llu\t0\t0\t%u\t%u\t%u\n", status ? k->name : demangled,
                (unsigned long long)k->start, (unsigned long long)k->end, k->deviceId, k->streamId, k->correlationId);
            std::free(demangled);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* m = (CUpti_ActivityMemcpy6*)record;
            std::fprintf(output, "memcpy\tcopy\t%llu\t%llu\t%llu\t%u\t%u\t%u\t%u\n",
                (unsigned long long)m->start, (unsigned long long)m->end, (unsigned long long)m->bytes,
                m->copyKind, m->deviceId, m->streamId, m->correlationId);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            auto* a = (CUpti_ActivityAPI*)record; const char* name = "unknown";
            cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
            std::fprintf(output, "api\t%s\t%llu\t%llu\t0\t0\t0\t%u\t%u\n", name,
                (unsigned long long)a->start, (unsigned long long)a->end, a->threadId, a->correlationId);
        }
    }
    size_t dropped = 0; cuptiActivityGetNumDroppedRecords(context, stream, &dropped);
    if (dropped) std::fprintf(output, "dropped\trecords\t0\t0\t%zu\t0\t0\t0\t0\n", dropped);
    std::free(buffer);
}
void gap_trace_begin(const char* name) {
    signal(SIGSEGV,crash);
    if (!initialized) { check(cuptiActivityRegisterCallbacks(requested, completed)); initialized = true; }
    std::string path = std::string(std::getenv("GAP_TRACE")) + "/" + name + ".tsv";
    output = std::fopen(path.c_str(), "w");
    if (!output) throw std::runtime_error("trace open failed");
    std::fprintf(output, "kind\tname\tstart\tend\tbytes\tcopy_kind\tdevice\tstream\tcorrelation\n");
    std::fflush(output);
    std::fprintf(stderr, "CUPTI enable begin %s\n", name);
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiGetTimestamp(&start));
    std::fprintf(stderr, "CUPTI enable done\n");
}
void gap_trace_end() {
    uint64_t end; check(cuptiGetTimestamp(&end));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityFlushAll(0));
    std::lock_guard<std::mutex> lock(mutex);
    std::fprintf(output, "window\twall\t%llu\t%llu\t0\t0\t0\t0\t0\n", (unsigned long long)start, (unsigned long long)end);
    std::fclose(output); output = nullptr;
}
