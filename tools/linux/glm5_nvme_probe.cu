// glm5_nvme_probe.cu — valida el mapa experto→LBA contra el SN5000S via gpu-nvme-direct.
// Requiere: Linux, VFIO bind del NVMe, driver nvidia parcheado (Tier 1).
// Build: nvcc -O2 -arch=sm_86 -I<gpunvme>/include glm5_nvme_probe.cu <gpunvme>/src/host/layer_loader.cu -o probe
// Uso: sudo ./probe <pci_bdf> <index_json>

#include "gpunvme/layer_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

// hashes del primer bloque de 3 slices (generados por build_lba_index.py en Windows,
// LBAs ya incluyen el offset de la particion = 2048 sectores)
struct Probe { int layer; const char* key; int expert; uint64_t lba; const char* hash16; };
static const Probe kProbes[] = {
    {23, "up",   77, 218125506ULL, "220df462cfd41245"},
    {44, "gate", 24, 371359667ULL, "d40d9dc76320007f"},
    {37, "up",   48, 340446352ULL, "e80ff6110c1dd0e7"},
};

static void hex16(const uint8_t* p, char* out) {
    for (int i = 0; i < 8; i++) sprintf(out + i * 2, "%02x", p[i]);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s <pci_bdf 01:00.0> <expert_lba_index.json>\n", argv[0]);
        return 1;
    }
    gpunvme_layer_loader_t loader;
    gpunvme_err_t e = gpunvme_layer_loader_init(&loader, argv[1], 1 << 20 /*max_layer*/, 8);
    if (e != GPUNVME_OK) {
        fprintf(stderr, "init failed: %d\n", e);
        return 1;
    }
    const uint32_t bs = gpunvme_layer_loader_block_size(&loader);
    fprintf(stderr, "probe: block_size=%u mdts=%u ns_blocks=%llu\n",
            bs, gpunvme_layer_loader_max_transfer(&loader),
            (unsigned long long)gpunvme_layer_loader_ns_blocks(&loader));

    // buffer pinned de 1MiB
    uint8_t* buf;
    if (cudaMallocHost((void**)&buf, 1 << 20) != cudaSuccess) {
        fprintf(stderr, "pin failed\n");
        return 1;
    }

    int fails = 0;
    for (const auto& p : kProbes) {
        // leer 4KB en el LBA del slice
        memset(buf, 0, 1 << 20);
        e = gpunvme_load_layer(&loader, p.lba, 4096, buf);
        if (e != GPUNVME_OK) {
            fprintf(stderr, "probe blk.%d.%s e=%d: read failed %d\n", p.layer, p.key, p.expert, e);
            fails++;
            continue;
        }
        char got[17] = {};
        hex16(buf, got);
        const bool ok = strncmp(got, p.hash16, 16) == 0;
        printf("blk.%d.%-4s e=%-3d lba=%-10llu %s (got %.16s want %.16s)\n",
               p.layer, p.key, p.expert, (unsigned long long)p.lba,
               ok ? "OK" : "FAIL", got, p.hash16);
        if (!ok) fails++;
    }

    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&loader);
    printf(fails ? "PROBE FAILED (%d)\n" : "PROBE OK — mapa LBA validado, el store es el archivo original\n", fails);
    return fails ? 1 : 0;
}
