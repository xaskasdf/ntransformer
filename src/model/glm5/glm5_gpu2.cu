// GLM-5.3-Flash decode fast-path: forward device-residente.
// Patrones rescatados de ter (D:\projects\ter): estado en device, escalas
// device-side, smem de tamano fijo, seleccion en-kernel, tabla de punteros.

#include "glm5_gpu2.h"
#include "glm5_gpu.h"
#include "../../core/types.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace nt {
namespace glm5 {
namespace {

constexpr int kThreads = 256;

// ---------------- kernels ----------------

__device__ __forceinline__ float dq8_sum32(const uint8_t* blk, const float* xb) {
    uint16_t dh;
    memcpy(&dh, blk, 2);
    const float d = __half2float(*reinterpret_cast<const __half*>(&dh));
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) s += (float)qs[i] * xb[i];
    return d * s;
}

// GEMV Q8_0 device->device, warp por fila
__global__ void k_q8(const uint8_t* __restrict__ W, const float* __restrict__ x,
                     float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dq8_sum32(row + (size_t)b * 34u, x + (size_t)b * 32u);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}

// GEMV F32 device->device
__global__ void k_f32v(const float* __restrict__ W, const float* __restrict__ x,
                       float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const float* row = W + (size_t)wid * cols;
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += row[c] * x[c];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[wid] = acc;
}

// rmsnorm in-place-ish: y = rms(x)*w ; single block, n <= 16384
__global__ void k_rmsn(const float* __restrict__ x, const float* __restrict__ w,
                       float* __restrict__ y, int n, float eps) {
    extern __shared__ float red[];   // 2*blockDim
    const int tid = threadIdx.x;
    float partial = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) partial += x[i] * x[i];
    red[tid] = partial;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// grouped strided Q8 (wkb/wvb): W[g] es [rows, cols], x [groups, cols] -> y [groups, rows]
__global__ void k_q8_gs(const uint8_t* __restrict__ W, const float* __restrict__ x,
                        float* __restrict__ y, int rows, int cols) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.y;
    if (wid >= rows) return;
    const int nb = cols >> 5;
    const uint8_t* Wg = W + (size_t)g * rows * nb * 34u;
    const uint8_t* row = Wg + (size_t)wid * nb * 34u;
    const float* xg = x + (size_t)g * cols;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dq8_sum32(row + (size_t)b * 34u, xg + (size_t)b * 32u);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)g * rows + wid] = acc;
}

// broadcast de emb a hc streams
__global__ void k_bcast(const float* __restrict__ x, float* __restrict__ streams, int H, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    const float v = x[i];
    for (int h = 0; h < hc; h++) streams[(size_t)h * H + i] = v;
}

// out += src
__global__ void k_add(float* __restrict__ out, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += src[i];
}

// layernorm con bias
__global__ void k_ln(const float* __restrict__ x, const float* __restrict__ w,
                     const float* __restrict__ b, float* __restrict__ y, int n, float eps) {
    extern __shared__ float red[];
    const int tid = threadIdx.x;
    float ps = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ps += x[i];
    red[tid] = ps;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    const float mean = red[0] / (float)n;
    float pv = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) pv += (x[i] - mean) * (x[i] - mean);
    red[tid] = pv;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    const float rstd = rsqrtf(red[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) y[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

// mHC: mixes[24] -> pre[4], post[4], comb[16]; un bloque
__global__ void k_hc_mix(const float* __restrict__ mixes, const float* __restrict__ base,
                         const float* __restrict__ scale, float eps, int iters,
                         float* __restrict__ pre, float* __restrict__ post,
                         float* __restrict__ comb) {
    // un solo hilo: 24 valores
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int hc = 4;
        for (int h = 0; h < hc; h++) {
            pre[h] = 1.0f / (1.0f + __expf(-(mixes[h] * scale[0] + base[h]))) + eps;
            post[h] = 2.0f / (1.0f + __expf(-(mixes[hc + h] * scale[1] + base[hc + h])));
        }
        float m[4][4];
        for (int dst = 0; dst < hc; dst++)
            for (int src = 0; src < hc; src++)
                m[dst][src] = mixes[2 * hc + dst + src * hc] * scale[2] + base[2 * hc + dst + src * hc];
        for (int src = 0; src < hc; src++) {
            float mx = m[0][src];
            for (int dst = 1; dst < hc; dst++) mx = fmaxf(mx, m[dst][src]);
            float sum = 0.0f;
            for (int dst = 0; dst < hc; dst++) { m[dst][src] = __expf(m[dst][src] - mx); sum += m[dst][src]; }
            for (int dst = 0; dst < hc; dst++) m[dst][src] /= sum;
        }
        for (int i = 0; i < 16; i++) ((float*)m)[i] += eps;
        auto nc = [&]() {
            for (int dst = 0; dst < hc; dst++) {
                float s = 0.0f;
                for (int src = 0; src < hc; src++) s += m[dst][src];
                s += eps;
                for (int src = 0; src < hc; src++) m[dst][src] /= s;
            }
        };
        auto nr = [&]() {
            for (int src = 0; src < hc; src++) {
                float s = 0.0f;
                for (int dst = 0; dst < hc; dst++) s += m[dst][src];
                s += eps;
                for (int dst = 0; dst < hc; dst++) m[dst][src] /= s;
            }
        };
        nc();
        for (int i = 1; i < iters; i++) { nr(); nc(); }
        for (int dst = 0; dst < hc; dst++)
            for (int src = 0; src < hc; src++)
                comb[dst + src * hc] = m[dst][src];
    }
}

// x[d] = sum_h streams[h*H+d] * pre[h]
__global__ void k_hc_collapse(const float* __restrict__ streams, const float* __restrict__ pre,
                              float* __restrict__ x, int H, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float acc = 0.0f;
    for (int h = 0; h < hc; h++) acc += streams[(size_t)h * H + i] * pre[h];
    x[i] = acc;
}

// out[dst*H+i] = branch[i]*post[dst] + sum_src res[src*H+i]*comb[dst+src*hc]
__global__ void k_hc_post(const float* __restrict__ branch, const float* __restrict__ res,
                          const float* __restrict__ post, const float* __restrict__ comb,
                          float* __restrict__ out, int H, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    for (int dst = 0; dst < hc; dst++) {
        float v = branch[i] * post[dst];
        for (int src = 0; src < hc; src++) v += res[(size_t)src * H + i] * comb[dst + src * hc];
        out[(size_t)dst * H + i] = v;
    }
}

// conv1d causal + silu, estado en device; grid = DI threads
__global__ void k_conv(const float* __restrict__ w,       // [DI][DC] (canal-major)
                       const float* __restrict__ src,
                       float* __restrict__ st,             // [P][DI]
                       float* __restrict__ dst, int DI, int P, int DC) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= DI) return;
    float acc = w[(size_t)c * DC + P] * src[c];
    for (int s = 0; s < P; s++) acc += w[(size_t)c * DC + s] * st[(size_t)s * DI + c];
    dst[c] = acc / (1.0f + __expf(-acc));
}

__global__ void k_conv_shift(float* __restrict__ st, const float* __restrict__ src, int DI, int P) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= DI) return;
    for (int s = 0; s + 1 < P; s++) st[(size_t)s * DI + c] = st[(size_t)(s + 1) * DI + c];
    st[(size_t)(P - 1) * DI + c] = src[c];
}

// gates: decay = exp(sigmoid(-(g1*dt+A... ))*lb); beta = sigmoid(beta)
__global__ void k_gdn_gates(const float* __restrict__ g1raw, const float* __restrict__ dtb,
                            const float* __restrict__ A, const float* __restrict__ betaraw,
                            float* __restrict__ decay, float* __restrict__ beta,
                            int DI, int HD, int NH, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= DI) return;
    const int h = i / HD;
    const float g = 1.0f / (1.0f + __expf((g1raw[i] + dtb[i]) * A[h])) * lb;   // sigmoid(-x) = 1/(1+e^x)
    decay[i] = __expf(g);
    if (i < NH) beta[i] = 1.0f / (1.0f + __expf(-betaraw[i]));
}

// GDN un bloque por cabeza: estado s[j*HD+i]=S[i][j]
__global__ void k_gdn(const float* __restrict__ q, const float* __restrict__ k,
                      const float* __restrict__ v, const float* __restrict__ dec,
                      const float* __restrict__ beta, float* __restrict__ S,
                      float* __restrict__ out, int HD) {
    extern __shared__ float sm[];    // 3*HD floats (k, q, dec)
    float* sk = sm;
    float* sq = sm + HD;
    float* sd = sm + 2 * HD;
    const int h = blockIdx.x;
    const int j = threadIdx.x;       // v-dim
    if (j >= HD) return;
    const float* kq = k + (size_t)h * HD;
    const float* qq = q + (size_t)h * HD;
    const float* vv = v + (size_t)h * HD;
    const float* dd = dec + (size_t)h * HD;
    for (int i = j; i < HD; i += blockDim.x) {
        sk[i] = kq[i];
        sq[i] = qq[i];
        sd[i] = dd[i];
    }
    __syncthreads();
    float* row = S + (size_t)h * HD * HD + (size_t)j * HD;
    float delta_j;
    {
        float dsum = 0.0f;
#pragma unroll 4
        for (int i = 0; i < HD; i++) {
            row[i] *= sd[i];
            dsum += row[i] * sk[i];
        }
        delta_j = (vv[j] - dsum) * beta[h];
    }
    __syncthreads();   // todas las filas aplicaron decay antes de la actualizacion
#pragma unroll 4
    for (int i = 0; i < HD; i++) row[i] += sk[i] * delta_j;
    float o = 0.0f;
#pragma unroll 4
    for (int i = 0; i < HD; i++) o += row[i] * sq[i];
    out[(size_t)h * HD + j] = o * rsqrtf((float)HD);
}

// out gate: rmsnorm por cabeza * sigmoid(g2); y l2norm opcional para q/k
__global__ void k_out_gate(float* __restrict__ o, const float* __restrict__ g2,
                           const float* __restrict__ w, int HD, int NH, float eps) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= HD) return;
    float* oh = o + (size_t)h * HD;
    __shared__ float red[kThreads];
    float partial = 0.0f;
    for (int i = j; i < HD; i += blockDim.x) partial += oh[i] * oh[i];
    red[j] = partial;
    __syncthreads();
    for (int o2 = blockDim.x >> 1; o2 > 0; o2 >>= 1) {
        if (j < o2) red[j] += red[j + o2];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)HD + eps);
    for (int i = j; i < HD; i += blockDim.x) oh[i] = oh[i] * inv * w[i];
    __syncthreads();
    oh[j] *= 1.0f / (1.0f + __expf(-g2[(size_t)h * HD + j]));
}

// l2norm por cabeza de un vector [NH*HD]
__global__ void k_l2heads(float* __restrict__ v, int HD, int NH, float eps) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= HD) return;
    float* vh = v + (size_t)h * HD;
    __shared__ float red[kThreads];
    float partial = 0.0f;
    for (int i = j; i < HD; i += blockDim.x) partial += vh[i] * vh[i];
    red[j] = partial;
    __syncthreads();
    for (int o2 = blockDim.x >> 1; o2 > 0; o2 >>= 1) {
        if (j < o2) red[j] += red[j + o2];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] + eps);
    for (int i = j; i < HD; i += blockDim.x) vh[i] *= inv;
}

// MLA attention sobre latentes [0..pos], todo visible (ctx<=2048):
// attn_lat[h][c] = sum_t softmax(q_abs_h . lat_t)[t] * lat_t[c]
__global__ void k_mla_attn(const float* __restrict__ q_abs,   // [NH*KL]
                           const float* __restrict__ lat,      // [(pos+1)*KL]
                           float* __restrict__ al,             // [NH*KL]
                           int pos, int KL, int NH) {
    extern __shared__ float sh[];    // scores[pos+1] (+ padding)
    const int h = blockIdx.x;
    const float* qa = q_abs + (size_t)h * KL;
    const float scale = rsqrtf((float)256.0f);
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float s = 0.0f;
        const float* row = lat + (size_t)t * KL;
#pragma unroll 8
        for (int c = 0; c < KL; c++) s += qa[c] * row[c];
        sh[t] = s * scale;
    }
    __syncthreads();
    float mx = sh[0];
    for (int t = 1; t <= pos; t++) mx = fmaxf(mx, sh[t]);
    float sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const float e = __expf(sh[t] - mx);
        sh[t] = e;
        sum += e;
    }
    __shared__ float red[1024];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    float* out = al + (size_t)h * KL;
    for (int c = threadIdx.x; c < KL; c += blockDim.x) out[c] = 0.0f;
    __syncthreads();
    for (int t = 0; t <= pos; t++) {
        const float wgt = sh[t] * inv;
        const float* row = lat + (size_t)t * KL;
        for (int c = threadIdx.x; c < KL; c += blockDim.x) out[c] += wgt * row[c];
    }
}

// pooling del indexer para un pool completado en pos: pos%4==3
__global__ void k_pool(const float* __restrict__ ig, const float* __restrict__ ik,
                       const float* __restrict__ ape,   // [kpool*IDX]
                       float* __restrict__ pooled, int IDX, int kpool) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= IDX) return;
    const int base = (blockIdx.y * kpool);
    float lg[8], pr[8];
    float mx = -INFINITY;
    for (int m = 0; m < kpool; m++) {
        lg[m] = ig[(size_t)(base + m) * IDX + d] + ape[(size_t)m * IDX + d];
        mx = fmaxf(mx, lg[m]);
    }
    float sum = 0.0f;
    for (int m = 0; m < kpool; m++) { pr[m] = __expf(lg[m] - mx); sum += pr[m]; }
    float acc = 0.0f;
    for (int m = 0; m < kpool; m++) acc += (pr[m] / sum) * ik[(size_t)(base + m) * IDX + d];
    pooled[(size_t)blockIdx.y * IDX + d] = acc;
}

// router: sigmoid + bias + top-8 (un bloque de >= 288 hilos)
__global__ void k_router(const float* __restrict__ logits, const float* __restrict__ bias,
                         float scale, int n_norm, int used,
                         int* __restrict__ ids, float* __restrict__ wts, int n_expert) {
    extern __shared__ float sc[];   // n_expert
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        sc[e] = 1.0f / (1.0f + __expf(-logits[e])) + bias[e];   // seleccion
    }
    __syncthreads();
    if (tid == 0) {
        // probs no sesgadas para pesos
        float probs[8];
        for (int k = 0; k < used; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int e = 0; e < n_expert; e++) {
                if (sc[e] > bv) { bv = sc[e]; best = e; }
            }
            ids[k] = best;
            probs[k] = 1.0f / (1.0f + __expf(-logits[best]));
            sc[best] = -INFINITY;
        }
        float wsum = 0.0f;
        for (int k = 0; k < used; k++) wsum += probs[k];
        if (n_norm) wsum = fmaxf(wsum, 6.103515625e-5f);
        for (int k = 0; k < used; k++) wts[k] = (n_norm ? probs[k] / wsum : probs[k]) * scale;
    }
}

// expertos: gate/up batched via tabla de punteros device
struct Ptr8 {
    const uint8_t* p[8];
};

__global__ void k_q8_tab_gu(const uint64_t* __restrict__ tab, const float* __restrict__ x,
                            float* __restrict__ yg, float* __restrict__ yu,
                            int rows, int cols, int n_used) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y >> 1;
    const bool is_up = blockIdx.y & 1;
    if (wid >= rows) return;
    const uint8_t* W = reinterpret_cast<const uint8_t*>(tab[(is_up ? n_used : 0) + k]);
    float* dst = (is_up ? yu : yg) + (size_t)k * rows;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dq8_sum32(row + (size_t)b * 34u, x + (size_t)b * 32u);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) dst[wid] = acc;
}

__global__ void k_q8_tab_down(const uint64_t* __restrict__ tab, const float* __restrict__ h,
                              float* __restrict__ yd, int rows, int cols, int n_used) {
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int k = blockIdx.y;
    if (wid >= rows) return;
    const uint8_t* W = reinterpret_cast<const uint8_t*>(tab[2 * n_used + k]);
    const float* hk = h + (size_t)k * cols;
    const int nb = cols >> 5;
    const uint8_t* row = W + (size_t)wid * nb * 34u;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) acc += dq8_sum32(row + (size_t)b * 34u, hk + (size_t)b * 32u);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) yd[(size_t)k * rows + wid] = acc;
}

__global__ void k_silu_clamp2(const float* __restrict__ g, const float* __restrict__ u,
                              float* __restrict__ h, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = fminf(g[i], limit);
    const float uv = fmaxf(-limit, fminf(u[i], limit));
    h[i] = gv / (1.0f + __expf(-gv)) * uv;
}

__global__ void k_expert_sum2(const float* __restrict__ wts, const float* __restrict__ down,
                              float* __restrict__ out, int n, int used) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        if (k < used) acc += wts[k] * down[(size_t)k * n + i];
    }
    out[i] = acc;
}

// media de streams + copia
__global__ void k_stream_mean(const float* __restrict__ streams, float* __restrict__ x, int H, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float acc = 0.0f;
    for (int h = 0; h < hc; h++) acc += streams[(size_t)h * H + i];
    x[i] = acc / (float)hc;
}

__global__ void k_copy_f32(const float* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---------------- host engine ----------------

struct Fast {
    FastConfig cfg;
    std::function<const void*(const std::string&)> lookup;
    cudaStream_t stream = nullptr;
    std::unordered_map<const void*, void*> res;
    std::mutex mtx;

    // device scratch
    float *d_x = nullptr, *d_streams = nullptr, *d_resid = nullptr;
    float *d_mixes = nullptr, *d_pre = nullptr, *d_post = nullptr, *d_comb = nullptr;
    float *d_a8 = nullptr, *d_b8 = nullptr, *d_c8 = nullptr;   // KDA buffers [DI]
    float *d_dec = nullptr, *d_beta = nullptr, *d_cout = nullptr;
    float *d_qr = nullptr, *d_q = nullptr, *d_kv = nullptr, *d_qabs = nullptr;
    float *d_ik = nullptr, *d_ig = nullptr, *d_iq = nullptr, *d_wp = nullptr;
    float *d_ape = nullptr, *d_al = nullptr, *d_attnout = nullptr;
    float *d_ff = nullptr, *d_h = nullptr;                      // [ff max]
    float *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
    float *d_idsf = nullptr;                                    // router logits [288]
    int   *d_ids = nullptr;
    float *d_wts = nullptr;
    uint64_t* d_etable = nullptr;                               // [24] expert ptrs
    float *d_logits = nullptr;
    float* pin_logits = nullptr;
    int*   pin_ids = nullptr;
    float* pin_wts = nullptr;
    float* pin_emb = nullptr;                                   // embedding staging
    float* d_flat = nullptr;                                    // hc*H normed flat
    float* d_g1 = nullptr, *d_g2 = nullptr, *d_fa = nullptr, *d_ga = nullptr;
    float* d_braw = nullptr;                                    // [n_heads]
    bool swap_conv = false;   // unused marker (conv outputs reuse FFN scratch)

    // per-layer state (device)
    float* d_conv[64] = {};      // [3 qkv][P][DI]  (per KDA layer, compact idx)
    float* d_ssm[64] = {};       // [NH*HD*HD]
    float* d_lat[16] = {};       // MLA latents [max_ctx*KL]
    float* d_idxk[16] = {};
    float* d_idxg[16] = {};
    float* d_pool[16] = {};

    int pos = 0;
    bool ready = false;
    uint64_t steps = 0;
    double seconds = 0;

    const void* W(const std::string& name) {
        const void* host = lookup(name);
        return host;
    }
    void* dev(const void* host, size_t bytes) {
        auto it = res.find(host);
        if (it != res.end()) return it->second;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        if (cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            return nullptr;
        }
        res.emplace(host, p);
        return p;
    }
    // resident por nombre, tamaño calculado por dims Q8/F32
    void* devq8(const std::string& name, int64_t rows, int64_t cols) {
        const void* host = lookup(name);
        if (!host) return nullptr;
        return dev(host, (size_t)rows * (cols / 32) * 34);
    }
    void* devf32(const std::string& name, int64_t n) {
        const void* host = lookup(name);
        if (!host) return nullptr;
        return dev(host, (size_t)n * 4);
    }

    bool init(const std::function<const void*(const std::string&)>& lk, const FastConfig& c) {
        lookup = lk;
        cfg = c;
        if (cudaStreamCreate(&stream) != cudaSuccess) return false;
        const int H = cfg.hidden, DI = cfg.n_heads * cfg.kda_head_dim;
        const int KL = cfg.kv_lora_rank;
        auto M = [&](float** p, size_t n) {
            return p && cudaMalloc(p, n * 4) == cudaSuccess;
        };
        if (cudaMalloc(&d_x, H * 4)) return false;
        if (cudaMalloc(&d_streams, (size_t)cfg.hc_mult * H * 4)) return false;
        if (cudaMalloc(&d_resid, (size_t)cfg.hc_mult * H * 4)) return false;
        if (cudaMalloc(&d_mixes, 24 * 4)) return false;
        if (cudaMalloc(&d_pre, 4 * 4) || cudaMalloc(&d_post, 4 * 4) || cudaMalloc(&d_comb, 16 * 4)) return false;
        if (cudaMalloc(&d_a8, DI * 4) || cudaMalloc(&d_b8, DI * 4) || cudaMalloc(&d_c8, DI * 4)) return false;
        if (cudaMalloc(&d_dec, DI * 4) || cudaMalloc(&d_beta, cfg.n_heads * 4) || cudaMalloc(&d_cout, DI * 4)) return false;
        if (cudaMalloc(&d_qr, cfg.q_lora_rank * 4)) return false;
        if (cudaMalloc(&d_q, (size_t)cfg.n_heads * cfg.qk_nope * 4)) return false;
        if (cudaMalloc(&d_kv, KL * 4)) return false;
        if (cudaMalloc(&d_qabs, (size_t)cfg.n_heads * KL * 4)) return false;
        if (cudaMalloc(&d_ik, cfg.indexer_head_dim * 4) || cudaMalloc(&d_ig, cfg.indexer_head_dim * 4)) return false;
        if (cudaMalloc(&d_iq, (size_t)cfg.indexer_n_head * cfg.indexer_head_dim * 4)) return false;
        if (cudaMalloc(&d_wp, cfg.indexer_n_head * 4)) return false;
        if (cudaMalloc(&d_ape, (size_t)cfg.indexer_kpool * cfg.indexer_head_dim * 4)) return false;
        if (cudaMalloc(&d_al, (size_t)cfg.n_heads * KL * 4)) return false;
        if (cudaMalloc(&d_attnout, (size_t)cfg.n_heads * cfg.v_dim * 4)) return false;
        const size_t ffmax = std::max<size_t>(std::max(cfg.n_ff_exp, cfg.n_ff_dense),
                                              (size_t)cfg.n_expert_used * cfg.n_ff_exp);
        if (cudaMalloc(&d_ff, ffmax * 4) || cudaMalloc(&d_h, ffmax * 4)) return false;
        if (cudaMalloc(&d_gate, ffmax * 4) || cudaMalloc(&d_up, ffmax * 4)) return false;
        if (cudaMalloc(&d_down, (size_t)cfg.n_expert_used * H * 4)) return false;
        if (cudaMalloc(&d_idsf, cfg.n_expert * 4)) return false;
        if (cudaMalloc(&d_ids, 8 * 4) || cudaMalloc(&d_wts, 8 * 4)) return false;
        if (cudaMalloc(&d_etable, 24 * sizeof(uint64_t))) return false;
        if (cudaMalloc(&d_logits, (size_t)cfg.vocab_size * 4)) return false;
        if (cudaMallocHost(&pin_logits, (size_t)cfg.vocab_size * 4)) return false;
        if (cudaMallocHost(&pin_ids, 8 * 4) || cudaMallocHost(&pin_wts, 8 * 4)) return false;
        if (cudaMallocHost(&pin_emb, H * 4)) return false;
        if (cudaMalloc(&d_flat, (size_t)cfg.hc_mult * H * 4)) return false;
        if (cudaMalloc(&d_g1, DI * 4) || cudaMalloc(&d_g2, DI * 4)) return false;
        if (cudaMalloc(&d_fa, cfg.kda_head_dim * 4) || cudaMalloc(&d_ga, cfg.kda_head_dim * 4)) return false;
        if (cudaMalloc(&d_braw, cfg.n_heads * 4)) return false;

        int rk = 0, rm = 0;
        for (int il = 0; il < cfg.n_layers; il++) {
            if (cfg.is_recr[il]) {
                if (cudaMalloc(&d_conv[rk], 3 * (cfg.d_conv - 1) * DI * 4)) return false;
                cudaMemset(d_conv[rk], 0, 3 * (cfg.d_conv - 1) * DI * 4);
                if (cudaMalloc(&d_ssm[rk], (size_t)cfg.n_heads * cfg.kda_head_dim * cfg.kda_head_dim * 4)) return false;
                cudaMemset(d_ssm[rk], 0, (size_t)cfg.n_heads * cfg.kda_head_dim * cfg.kda_head_dim * 4);
                rk++;
            } else {
                if (cudaMalloc(&d_lat[rm], (size_t)cfg.max_ctx * KL * 4)) return false;
                if (cudaMalloc(&d_idxk[rm], (size_t)cfg.max_ctx * cfg.indexer_head_dim * 4)) return false;
                if (cudaMalloc(&d_idxg[rm], (size_t)cfg.max_ctx * cfg.indexer_head_dim * 4)) return false;
                if (cudaMalloc(&d_pool[rm], (size_t)(cfg.max_ctx / cfg.indexer_kpool + 2) * cfg.indexer_head_dim * 4)) return false;
                cudaMemset(d_pool[rm], 0, (size_t)(cfg.max_ctx / cfg.indexer_kpool + 2) * cfg.indexer_head_dim * 4);
                rm++;
            }
        }
        ready = true;
        return true;
    }

    ~Fast() {
        for (auto& kv : res) cudaFree(kv.second);
        cudaFree(d_x); cudaFree(d_streams); cudaFree(d_resid);
        cudaFree(d_mixes); cudaFree(d_pre); cudaFree(d_post); cudaFree(d_comb);
        cudaFree(d_a8); cudaFree(d_b8); cudaFree(d_c8);
        cudaFree(d_dec); cudaFree(d_beta); cudaFree(d_cout);
        cudaFree(d_qr); cudaFree(d_q); cudaFree(d_kv); cudaFree(d_qabs);
        cudaFree(d_ik); cudaFree(d_ig); cudaFree(d_iq); cudaFree(d_wp); cudaFree(d_ape);
        cudaFree(d_al); cudaFree(d_attnout); cudaFree(d_ff); cudaFree(d_h);
        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_down);
        cudaFree(d_idsf); cudaFree(d_ids); cudaFree(d_wts); cudaFree(d_etable);
        cudaFree(d_logits);
        if (pin_logits) cudaFreeHost(pin_logits);
        if (pin_ids) cudaFreeHost(pin_ids);
        if (pin_wts) cudaFreeHost(pin_wts);
        for (int i = 0; i < 64; i++) { if (d_conv[i]) cudaFree(d_conv[i]); if (d_ssm[i]) cudaFree(d_ssm[i]); }
        for (int i = 0; i < 16; i++) {
            if (d_lat[i]) cudaFree(d_lat[i]);
            if (d_idxk[i]) cudaFree(d_idxk[i]);
            if (d_idxg[i]) cudaFree(d_idxg[i]);
            if (d_pool[i]) cudaFree(d_pool[i]);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

Fast& fast() {
    static Fast inst;
    return inst;
}

} // namespace

bool glm5_fast_init(const std::function<const void*(const std::string&)>& lookup,
                    const FastConfig& cfg) {
    return fast().init(lookup, cfg);
}

void glm5_fast_reset() {
    Fast& f = fast();
    if (!f.ready) return;
    const int DI = f.cfg.n_heads * f.cfg.kda_head_dim;
    int rk = 0, rm = 0;
    for (int il = 0; il < f.cfg.n_layers; il++) {
        if (f.cfg.is_recr[il]) {
            cudaMemset(f.d_conv[rk], 0, 3 * (f.cfg.d_conv - 1) * DI * 4);
            cudaMemset(f.d_ssm[rk], 0, (size_t)f.cfg.n_heads * f.cfg.kda_head_dim * f.cfg.kda_head_dim * 4);
            rk++;
        } else rm++;
    }
    f.pos = 0;
}

bool glm5_fast_ready() { return fast().ready; }

void glm5_fast_stats() {
    Fast& f = fast();
    fprintf(stderr, "[glm5-fast] steps=%llu seconds=%.1f residents=%zu tensors\n",
            (unsigned long long)f.steps, f.seconds, f.res.size());
}

bool glm5_fast_step(int token, float* logits) {
    Fast& f = fast();
    if (!f.ready) return false;
    static const bool dbg = getenv("GLM5_DBG_FAST") != nullptr;
    if (dbg) fprintf(stderr, "[fast] step token=%d pos=%d\n", token, f.pos);
    const FastConfig& c = f.cfg;
    const int H = c.hidden, DI = c.n_heads * c.kda_head_dim, HD = c.kda_head_dim, NH = c.n_heads;
    const int KL = c.kv_lora_rank, QK = c.qk_nope, VD = c.v_dim, hc = c.hc_mult;
    cudaStream_t s = f.stream;
    const int pos = f.pos;
    const auto t0 = std::chrono::steady_clock::now();

    auto G = [](int64_t n) { return (unsigned)((n + kThreads - 1) / kThreads); };
    auto gemv_q8 = [&](const std::string& name, const float* x, float* y, int64_t rows, int64_t cols) {
        void* w = f.devq8(name, rows, cols);
        if (!w) { fprintf(stderr, "fast: missing %s\n", name.c_str()); return false; }
        k_q8<<<G(rows * 32), kThreads, 0, s>>>((const uint8_t*)w, x, y, (int)rows, (int)cols);
        return true;
    };
    auto gemv_f32 = [&](const std::string& name, const float* x, float* y, int64_t rows, int64_t cols) {
        void* w = f.devf32(name, rows * cols);
        if (!w) { fprintf(stderr, "fast: missing %s\n", name.c_str()); return false; }
        k_f32v<<<G(rows * 32), kThreads, 0, s>>>((const float*)w, x, y, (int)rows, (int)cols);
        return true;
    };
    auto rmsn = [&](const std::string& wname, float* x, int n, float* y) {
        const float* w = wname.empty() ? nullptr : (const float*)f.devf32(wname, n);
        const int threads = std::min(1024, std::max(64, n));
        const size_t smem = 2 * threads * 4;
        k_rmsn<<<1, threads, smem, s>>>(x, w, y ? y : x, n, c.norm_eps);
    };
    auto mhc = [&](const std::string& kind, int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        // flat rmsnorm de los streams
        k_rmsn<<<1, 1024, 2 * 1024 * 4, s>>>(f.d_streams, nullptr, f.d_flat, hc * H, c.norm_eps);
        if (!gemv_q8(p + "hc_" + kind + "_fn.weight", f.d_flat, f.d_mixes, 24, (int64_t)hc * H)) return false;
        const float* base = (const float*)f.devf32(p + "hc_" + kind + "_base.weight", 24);
        const float* scale = (const float*)f.devf32(p + "hc_" + kind + "_scale.weight", 3);
        k_hc_mix<<<1, 32, 0, s>>>(f.d_mixes, base, scale, c.hc_eps, c.hc_sinkhorn_iters,
                                  f.d_pre, f.d_post, f.d_comb);
        k_hc_collapse<<<G(H), kThreads, 0, s>>>(f.d_streams, f.d_pre, f.d_x, H, hc);
        const float* nw = (const float*)f.devf32(p + kind + "_norm.weight", H);
        k_rmsn<<<1, 1024, 2 * 1024 * 4, s>>>(f.d_x, nw, f.d_x, H, c.norm_eps);
        return true;
    };

    auto dump = [&](const char* tag, const float* dev, int n) {
        if (!dbg) return;
        static float* pin = [] { float* p = nullptr; cudaMallocHost((void**)&p, 200000 * 4); return p; }();
        cudaMemcpy(pin, dev, n * 4, cudaMemcpyDeviceToHost);
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)pin[i] * pin[i];
        fprintf(stderr, "[fast-cp] %s n=%d ss=%.6f head: %.6f %.6f %.6f\n", tag, n, ss, pin[0], pin[1], pin[2]);
    };

    // ---- embedding ----
    {
        const uint8_t* emb = (const uint8_t*)f.lookup("token_embd.weight");
        if (!emb) return false;
        const int nb = H / 32;
        const uint8_t* row = emb + (size_t)token * nb * 34;
        for (int b = 0; b < nb; b++) {
            uint16_t dh;
            memcpy(&dh, row + (size_t)b * 34, 2);
            const float d = fp16_to_fp32_host(dh);
            const int8_t* qs = (const int8_t*)(row + (size_t)b * 34 + 2);
            for (int i = 0; i < 32; i++) f.pin_emb[b * 32 + i] = d * (float)qs[i];
        }
        cudaMemcpyAsync(f.d_x, f.pin_emb, H * 4, cudaMemcpyHostToDevice, s);
        k_bcast<<<G(H), kThreads, 0, s>>>(f.d_x, f.d_streams, H, hc);
    }

    int rk = 0, rm = 0;
    for (int il = 0; il < c.n_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const int cs = rk;

        // ---- attention sublayer ----
        if (dbg) fprintf(stderr, "[fast] il=%d attn\n", il);
        k_copy_f32<<<G((int64_t)hc * H), kThreads, 0, s>>>(f.d_streams, f.d_resid, hc * H);
        if (!mhc("attn", il)) return false;
        if (dbg && pos == 0 && il == 0) dump("attn_norm", f.d_x, H);
        if (dbg) fprintf(stderr, "[fast] il=%d mhc-attn done\n", il);

        if (c.is_recr[il]) {
            if (!gemv_q8(p + "attn_q.weight", f.d_x, f.d_a8, DI, H)) return false;
            if (!gemv_q8(p + "attn_k.weight", f.d_x, f.d_b8, DI, H)) return false;
            if (!gemv_q8(p + "attn_v.weight", f.d_x, f.d_c8, DI, H)) return false;
            if (!gemv_q8(p + "ssm_f_a.weight", f.d_x, f.d_fa, HD, H)) return false;
            if (!gemv_q8(p + "ssm_g_a.weight", f.d_x, f.d_ga, HD, H)) return false;
            if (!gemv_q8(p + "ssm_beta.weight", f.d_x, f.d_braw, NH, H)) return false;
            // conv: salida en buffers separados; el shift guarda la proyeccion cruda
            {
                const int P = c.d_conv - 1, DC = c.d_conv;
                const char* cn[3] = { "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight" };
                const float* srcs[3] = { f.d_a8, f.d_b8, f.d_c8 };
                float* dsts[3] = { f.d_gate, f.d_up, f.d_h };   // reuse FFN scratch (libre en atencion)
                for (int qkv = 0; qkv < 3; qkv++) {
                    const float* w = (const float*)f.devf32(p + cn[qkv], DI * DC);
                    float* st = f.d_conv[cs] + (size_t)qkv * P * DI;
                    k_conv<<<G(DI), kThreads, 0, s>>>(w, srcs[qkv], st, dsts[qkv], DI, P, DC);
                }
                for (int qkv = 0; qkv < 3; qkv++) {
                    float* st = f.d_conv[cs] + (size_t)qkv * P * DI;
                    k_conv_shift<<<G(DI), kThreads, 0, s>>>(st, srcs[qkv], DI, P);
                }
                // l2norm + gdn leen las salidas de conv (d_gate/d_up/d_h)
            }
            // gates + gdn
            if (!gemv_q8(p + "ssm_f_b.weight", f.d_fa, f.d_g1, DI, HD)) return false;
            if (!gemv_q8(p + "ssm_g_b.weight", f.d_ga, f.d_g2, DI, HD)) return false;
            {
                const float* dtb = (const float*)f.devf32(p + "ssm_dt.bias", DI);
                const float* A = (const float*)f.devf32(p + "ssm_a", NH);
                k_gdn_gates<<<G(DI), kThreads, 0, s>>>(f.d_g1, dtb, A, f.d_braw,
                                                       f.d_dec, f.d_beta, DI, HD, NH, c.kda_gate_lower_bound);
            }
            k_l2heads<<<NH, HD, 0, s>>>(f.d_gate, HD, NH, 1e-6f);
            k_l2heads<<<NH, HD, 0, s>>>(f.d_up, HD, NH, 1e-6f);
            k_gdn<<<NH, HD, 3 * HD * 4, s>>>(f.d_gate, f.d_up, f.d_h, f.d_dec, f.d_beta,
                                             f.d_ssm[cs], f.d_cout, HD);            {
                const float* ow = (const float*)f.devf32(p + "ssm_norm.weight", HD);
                k_out_gate<<<NH, HD, 0, s>>>(f.d_cout, f.d_g2, ow, HD, NH, c.norm_eps);
            }
            if (!gemv_q8(p + "attn_output.weight", f.d_cout, f.d_x, H, DI)) return false;
            if (dbg && pos == 0 && il == 0) dump("kda_out", f.d_x, H);
            rk++;
        } else {
            if (!gemv_q8(p + "attn_q_a.weight", f.d_x, f.d_qr, c.q_lora_rank, H)) return false;
            rmsn(p + "attn_q_a_norm.weight", f.d_qr, c.q_lora_rank, nullptr);
            if (!gemv_q8(p + "attn_q_b.weight", f.d_qr, f.d_q, (int64_t)NH * QK, c.q_lora_rank)) return false;
            if (!gemv_q8(p + "attn_kv_a_mqa.weight", f.d_x, f.d_kv, KL, H)) return false;
            rmsn(p + "attn_kv_a_norm.weight", f.d_kv, KL, nullptr);
            k_copy_f32<<<G(KL), kThreads, 0, s>>>(f.d_kv, f.d_lat[rm] + (size_t)pos * KL, KL);
            {
                void* wkb = f.devq8(p + "attn_k_b.weight", (int64_t)NH * KL, QK);
                if (!wkb) return false;
                dim3 grid((unsigned)((KL * 32 + kThreads - 1) / kThreads), NH);
                k_q8_gs<<<grid, kThreads, 0, s>>>((const uint8_t*)wkb, f.d_q, f.d_qabs, KL, QK);
            }
            k_mla_attn<<<NH, 512, (c.max_ctx + 1) * 4, s>>>(f.d_qabs, f.d_lat[rm], f.d_al, pos, KL, NH);
            {
                void* wvb = f.devq8(p + "attn_v_b.weight", (int64_t)NH * VD, KL);
                if (!wvb) return false;
                dim3 grid((unsigned)((VD * 32 + kThreads - 1) / kThreads), NH);
                k_q8_gs<<<grid, kThreads, 0, s>>>((const uint8_t*)wvb, f.d_al, f.d_attnout, VD, KL);
            }
            if (!gemv_q8(p + "attn_output.weight", f.d_attnout, f.d_x, H, (int64_t)NH * VD)) return false;
            if (dbg && pos <= 1 && il == 3) dump("mla_out", f.d_x, H);
            rm++;
        }

        // post-mix atencion
        k_hc_post<<<G(H), kThreads, 0, s>>>(f.d_x, f.d_resid, f.d_post, f.d_comb, f.d_streams, H, hc);

        // ---- FFN sublayer ----
        k_copy_f32<<<G((int64_t)hc * H), kThreads, 0, s>>>(f.d_streams, f.d_resid, hc * H);
        if (!mhc("ffn", il)) return false;

        if (il < c.dense_lead) {
            if (!gemv_q8(p + "ffn_gate.weight", f.d_x, f.d_gate, c.n_ff_dense, H)) return false;
            if (!gemv_q8(p + "ffn_up.weight", f.d_x, f.d_up, c.n_ff_dense, H)) return false;
            k_silu_clamp2<<<G((int64_t)c.n_ff_dense), kThreads, 0, s>>>(
                f.d_gate, f.d_up, f.d_h, c.n_ff_dense, 10.0f);
            if (!gemv_q8(p + "ffn_down.weight", f.d_h, f.d_x, H, c.n_ff_dense)) return false;
        } else {
            // router on-device
            if (dbg && pos <= 1 && il == 3) dump("moe_in", f.d_x, H);
            if (!gemv_f32(p + "ffn_gate_inp.weight", f.d_x, f.d_idsf, c.n_expert, H)) return false;
            {
                const float* bias = (const float*)f.devf32(p + "exp_probs_b.bias", c.n_expert);
                k_router<<<1, 288, c.n_expert * 4, s>>>(f.d_idsf, bias, c.expert_weights_scale,
                                                        c.expert_weights_norm ? 1 : 0, c.n_expert_used,
                                                        f.d_ids, f.d_wts, c.n_expert);
            }
            // host: leer ids, preparar slots, tabla de punteros
            if (dbg) fprintf(stderr, "[fast] il=%d router wait\n", il);
            cudaMemcpyAsync(f.pin_ids, f.d_ids, 8 * 4, cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(f.pin_wts, f.d_wts, 8 * 4, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            if (dbg && pos <= 1) {
                fprintf(stderr, "[fast-ids] t=%d il=%d:", pos, il);
                for (int k = 0; k < c.n_expert_used; k++) {
                    fprintf(stderr, " %d(%.4f)", f.pin_ids[k], f.pin_wts[k]);
                }
                fprintf(stderr, "\n");
            }
            if (dbg) fprintf(stderr, "[fast] il=%d staging experts\n", il);
            if (!glm5_gpu_stage_experts(f.lookup(p + "ffn_gate_exps.weight"),
                                        f.lookup(p + "ffn_up_exps.weight"),
                                        f.lookup(p + "ffn_down_exps.weight"),
                                        c.n_ff_exp, f.pin_ids, c.n_expert_used,
                                        f.d_etable, (void*)s)) {
                return false;
            }
            const float lim = 10.0f;
            {
                dim3 grid((unsigned)((c.n_ff_exp * 32 + kThreads - 1) / kThreads), 2 * c.n_expert_used);
                k_q8_tab_gu<<<grid, kThreads, 0, s>>>(f.d_etable, f.d_x, f.d_gate, f.d_up,
                                                      c.n_ff_exp, H, c.n_expert_used);
            }
            if (dbg && pos == 0 && il == 3) {
                // referencia host completa del gate del experto 0
                cudaMemcpy(f.pin_emb, f.d_gate, 8 * 4, cudaMemcpyDeviceToHost);
                float* xh = new float[H];
                cudaMemcpy(xh, f.d_x, H * 4, cudaMemcpyDeviceToHost);
                const uint8_t* gw = (const uint8_t*)f.lookup(p + "ffn_gate_exps.weight");
                const int nb = H / 32;
                const size_t gate_slice = (size_t)c.n_ff_exp * nb * 34;
                const uint8_t* row = gw + (size_t)f.pin_ids[0] * gate_slice;
                float ref = 0.0f;
                for (int b = 0; b < nb; b++) {
                    uint16_t dh;
                    memcpy(&dh, row + (size_t)b * 34, 2);
                    const float d = fp16_to_fp32_host(dh);
                    const int8_t* qs = (const int8_t*)(row + (size_t)b * 34 + 2);
                    float sx = 0.0f;
                    for (int i = 0; i < 32; i++) sx += (float)qs[i] * xh[b * 32 + i];
                    ref += d * sx;
                }
                delete[] xh;
                fprintf(stderr, "[fast-cp] gate0 dev=%.6f ref=%.6f ids0=%d\n",
                        f.pin_emb[0], ref, f.pin_ids[0]);
            }
            k_silu_clamp2<<<G((int64_t)c.n_ff_exp * c.n_expert_used), kThreads, 0, s>>>(
                f.d_gate, f.d_up, f.d_h, c.n_ff_exp * c.n_expert_used, lim);
            if (dbg && pos == 0 && il == 3) {
                float* hh = new float[c.n_ff_exp];
                float* gu = new float[2 * c.n_ff_exp];
                cudaMemcpy(gu, f.d_gate, c.n_ff_exp * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(gu + c.n_ff_exp, f.d_up, c.n_ff_exp * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hh, f.d_h, c.n_ff_exp * 4, cudaMemcpyDeviceToHost);
                float h0ref = fminf(gu[0], 10.0f) / (1.0f + expf(-fminf(gu[0], 10.0f))) *
                                fmaxf(-10.0f, fminf(gu[c.n_ff_exp], 10.0f));
                fprintf(stderr, "[fast-cp] h0 dev=%.6f ref=%.6f (g0=%.4f u0=%.4f)\n",
                        hh[0], h0ref, gu[0], gu[c.n_ff_exp]);
                delete[] hh; delete[] gu;
            }
            // guardar el input del MoE (d_x lo sobrescribe el routed sum)
            k_copy_f32<<<G(H), kThreads, 0, s>>>(f.d_x, f.d_flat, H);
            {
                dim3 grid((unsigned)((H * 32 + kThreads - 1) / kThreads), c.n_expert_used);
                k_q8_tab_down<<<grid, kThreads, 0, s>>>(f.d_etable, f.d_h, f.d_down,
                                                        H, c.n_ff_exp, c.n_expert_used);
            }
            if (dbg && pos == 0 && il == 3) {
                float* hbuf = new float[c.n_ff_exp];
                float d0 = 0.0f;
                cudaMemcpy(&d0, f.d_down, 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(hbuf, f.d_h, c.n_ff_exp * 4, cudaMemcpyDeviceToHost);
                const uint8_t* dw8 = (const uint8_t*)f.lookup(p + "ffn_down_exps.weight");
                const size_t down_slice = (size_t)H * (c.n_ff_exp / 32) * 34;
                const uint8_t* row = dw8 + (size_t)f.pin_ids[0] * down_slice;
                float ref = 0.0f;
                const int nb = c.n_ff_exp / 32;
                for (int b = 0; b < nb; b++) {
                    uint16_t dh2;
                    memcpy(&dh2, row + (size_t)b * 34, 2);
                    const float d = fp16_to_fp32_host(dh2);
                    const int8_t* qs = (const int8_t*)(row + (size_t)b * 34 + 2);
                    float sx = 0.0f;
                    for (int i = 0; i < 32; i++) sx += (float)qs[i] * hbuf[b * 32 + i];
                    ref += d * sx;
                }
                fprintf(stderr, "[fast-cp] down0 dev=%.6f ref=%.6f\n", d0, ref);
                delete[] hbuf;
            }
            k_expert_sum2<<<G(H), kThreads, 0, s>>>(f.d_wts, f.d_down, f.d_x, H, c.n_expert_used);
            if (dbg && pos <= 1 && il == 3) dump("moe_routed", f.d_x, H);
            // shared expert (con el input original guardado en d_flat)
            if (!gemv_q8(p + "ffn_gate_shexp.weight", f.d_flat, f.d_gate, c.n_ff_exp, H)) return false;
            if (!gemv_q8(p + "ffn_up_shexp.weight", f.d_flat, f.d_up, c.n_ff_exp, H)) return false;
            k_silu_clamp2<<<G((int64_t)c.n_ff_exp), kThreads, 0, s>>>(
                f.d_gate, f.d_up, f.d_h, c.n_ff_exp, 10.0f);
            if (!gemv_q8(p + "ffn_down_shexp.weight", f.d_h, f.d_ff, H, c.n_ff_exp)) return false;
            if (dbg && pos == 0 && il == 3) dump("shared", f.d_ff, H);
            k_add<<<G(H), kThreads, 0, s>>>(f.d_x, f.d_ff, H);
        }

        // post-mix FFN
        k_hc_post<<<G(H), kThreads, 0, s>>>(f.d_x, f.d_resid, f.d_post, f.d_comb, f.d_streams, H, hc);

        if (dbg) {
            static float* pin_dbg = [] {
                float* p = nullptr;
                cudaMallocHost((void**)&p, 4 * 4096 * 4);
                return p;
            }();
            cudaMemcpy(pin_dbg, f.d_streams, (size_t)hc * H * 4, cudaMemcpyDeviceToHost);
            double ss = 0;
            for (int i = 0; i < hc * H; i++) ss += (double)pin_dbg[i] * pin_dbg[i];
            fprintf(stderr, "[fast-dbg] t=%d il=%-2d ss=%.6f head: %.5f %.5f %.5f %.5f\n",
                    pos, il, ss, pin_dbg[0], pin_dbg[1], pin_dbg[2], pin_dbg[3]);
        }
    }

    // ---- terminal ----
    k_stream_mean<<<G(H), kThreads, 0, s>>>(f.d_streams, f.d_x, H, hc);
    rmsn("output_norm.weight", f.d_x, H, nullptr);
    if (!gemv_q8("output.weight", f.d_x, f.d_logits, c.vocab_size, H)) return false;
    cudaMemcpyAsync(f.pin_logits, f.d_logits, (size_t)c.vocab_size * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    memcpy(logits, f.pin_logits, (size_t)c.vocab_size * 4);

    f.pos++;
    ++f.steps;
    f.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

} // namespace glm5
} // namespace nt
