#include "trace.h"
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <cstdio>
static void crash(int){void* f[40];int n=backtrace(f,40);backtrace_symbols_fd(f,n,2);_exit(139);}
int main(){signal(SIGSEGV,crash);uint32_t version;cuptiGetVersion(&version);std::fprintf(stderr,"CUPTI runtime=%u headers=%u\n",version,CUPTI_API_VERSION);void* p;cudaMalloc(&p,4096);gap_trace_begin("smoke");cudaMemset(p,0,4096);cudaDeviceSynchronize();gap_trace_end();cudaFree(p);}
