#include <cuda_runtime.h>
#include <cstdint>
__global__ void probe(float2* out, const float2* x, const float2* f, int64_t n, int mode) {
    int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    float a=x[i].x, b=x[i].y, c=f[i].x, d=f[i].y;
    if (mode == 4) out[i] = make_float2(a*c-b*d, a*d+b*c);
    else out[i] = make_float2((mode&1) ? fmaf(-b,d,a*c) : fmaf(a,c,-b*d),
                             (mode&2) ? fmaf(b,c,a*d) : fmaf(a,d,b*c));
}
extern "C" int run_probe(void* out, const void* x, const void* f, int64_t n, int mode, void* stream) {
    probe<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<float2*>(out),static_cast<const float2*>(x),static_cast<const float2*>(f),n,mode);
    return cudaGetLastError();
}
