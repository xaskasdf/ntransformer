// glm5_nvme_probe2.cu — valida el índice LBA regenerado (lba0 + in_off) por el
// camino GPU→NVMe. Lee 4KB alineados en lba0 y compara [in_off..in_off+16].
// Build: nvcc -O2 -arch=sm_86 -I<gpunvme>/include glm5_nvme_probe2.cu -L<build-hw> \
//          -lgpunvme_layer_loader -lgpunvme_host -lgpunvme_device -o probe2
// Uso: sudo ./probe2 <pci_bdf con dominio> <probe_vectors.json>
#include "gpunvme/layer_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct Vec { int layer, expert; uint64_t lba0; int in_off; char key[8]; uint8_t want[16]; };

static std::string slurp(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return "";
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, 0); size_t got = fread(&s[0], 1, n, f); (void)got; fclose(f); return s;
}
static long find_num(const std::string& s, const char* key, long from) {
    long p = (long)s.find(key, from); if (p < 0) return -1;
    return strtoull(s.c_str() + p + strlen(key), nullptr, 10);
}
static void find_str(const std::string& s, const char* key, long from, char* out, int cap) {
    long p = (long)s.find(key, from);
    if (p < 0) { out[0] = 0; return; }
    const char* q = s.c_str() + p + strlen(key);
    int i = 0; while (*q && *q != '"' && i < cap - 1) out[i++] = *q++;
    out[i] = 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <bdf> <probe_vectors.json>\n", argv[0]); return 1; }
    std::string js = slurp(argv[2]);
    if (js.empty()) { fprintf(stderr, "no se pudo leer %s\n", argv[2]); return 1; }

    std::vector<Vec> vecs;
    long pos = 0;
    while ((pos = (long)js.find("\"layer\"", pos + 1)) > 0) {
        Vec v; memset(&v, 0, sizeof v);
        v.layer = (int)find_num(js, "\"layer\": ", pos);
        find_str(js, "\"key\": \"", pos, v.key, sizeof v.key);
        v.expert = (int)find_num(js, "\"expert\": ", pos);
        v.lba0 = find_num(js, "\"lba0\": ", pos);
        v.in_off = (int)find_num(js, "\"in_off\": ", pos);
        char hex[64]; find_str(js, "\"want\": \"", pos, hex, sizeof hex);
        for (int i = 0; i < 16; i++) {
            unsigned x; sscanf(hex + i * 2, "%2x", &x); v.want[i] = (uint8_t)x;
        }
        vecs.push_back(v);
    }
    fprintf(stderr, "probe2: %zu vectores\n", vecs.size());
    if (vecs.empty()) return 1;

    gpunvme_layer_loader_t loader;
    gpunvme_err_t e = gpunvme_layer_loader_init(&loader, argv[1], 1 << 20, 8);
    if (e != GPUNVME_OK) { fprintf(stderr, "init failed: %d\n", e); return 1; }

    uint8_t* buf;
    if (cudaMallocHost((void**)&buf, 1 << 20) != cudaSuccess) { fprintf(stderr, "pin failed\n"); return 1; }

    int fails = 0;
    for (size_t i = 0; i < vecs.size(); i++) {
        auto& v = vecs[i];
        memset(buf, 0, 4096);
        e = gpunvme_load_layer(&loader, v.lba0, 4096, buf);
        bool ok = (e == GPUNVME_OK) && (memcmp(buf + v.in_off, v.want, 16) == 0);
        printf("blk%02d.%-4s e=%-3d lba0=%-10llu in_off=%-3d %s",
               v.layer, v.key, v.expert, (unsigned long long)v.lba0, v.in_off, ok ? "OK" : "FAIL");
        if (!ok) { printf(" (e=%d got %.16s want %.16s)", e, buf + v.in_off, v.want); fails++; }
        printf("\n");
    }
    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&loader);
    if (fails) printf("PROBE2 FAILED (%d/%zu)\n", fails, vecs.size());
    else printf("PROBE2 OK — %zu/%zu slices validados por GPU->NVMe (lba0+in_off)\n", vecs.size(), vecs.size());
    return fails ? 1 : 0;
}
