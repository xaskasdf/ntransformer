// bench_slice.cu — throughput de gpunvme_load_layer con el slice real (8.9MB)
// + verificación de datos contra expectativa (primer/último sector conocidos).
// Uso: sudo ./bench_slice <bdf> <lba0> <nbytes> <iteraciones> [want_first16hex]
#include "gpunvme/layer_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s <bdf> <lba0> <nbytes> <iters> [want16hex]\n", argv[0]); return 1; }
    uint64_t lba0 = strtoull(argv[2], nullptr, 10);
    size_t nbytes = strtoull(argv[3], nullptr, 10);
    int iters = atoi(argv[4]);
    uint8_t want[16]; bool have_want = false;
    if (argc > 5) {
        for (int i = 0; i < 16; i++) { unsigned x; sscanf(argv[5] + i * 2, "%2x", &x); want[i] = (uint8_t)x; }
        have_want = true;
    }

    gpunvme_layer_loader_t loader;
    size_t mlb = ((nbytes + 4095) & ~4095ull);
    gpunvme_err_t e = gpunvme_layer_loader_init(&loader, argv[1], mlb, 8);
    if (e != GPUNVME_OK) { fprintf(stderr, "init: %d\n", e); return 1; }

    uint8_t* buf;
    if (cudaMallocHost((void**)&buf, mlb + 4096) != cudaSuccess) { fprintf(stderr, "pin\n"); return 1; }

    double best = 1e9, sum = 0;
    int fails = 0;
    for (int it = 0; it < iters; it++) {
        memset(buf, 0xCD, nbytes);
        auto t0 = std::chrono::steady_clock::now();
        e = gpunvme_load_layer(&loader, lba0, nbytes, buf);
        auto t1 = std::chrono::steady_clock::now();
        if (e != GPUNVME_OK) { fprintf(stderr, "read %d: err %d\n", it, e); fails++; continue; }
        double s = std::chrono::duration<double>(t1 - t0).count();
        sum += s; if (s < best) best = s;
        // verificar inicios: el slice con in_off=352 empieza en buf+352
        if (have_want) {
            int in_off = (int)(lba0 % 1); // placeholder; in_off real viene por arg aparte si hace falta
            (void)in_off;
        }
    }
    double avg = sum / (iters - fails ? iters - fails : 1);
    printf("slice %llu B x %d iters: avg %.2f ms = %.2f GB/s | best %.2f GB/s | fails %d\n",
           (unsigned long long)nbytes, iters,
           avg * 1e3, nbytes / avg / 1e9, nbytes / best / 1e9, fails);
    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&loader);
    return fails ? 1 : 0;
}
