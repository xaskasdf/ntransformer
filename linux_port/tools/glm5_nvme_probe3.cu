// glm5_nvme_probe3.cu — valida el FILL COMPLETO de slices (runs multi + single)
// leyendo cada slice entero como lo hace el ExpertEngine y comparando a offsets
// interiores contra los bytes reales del store (pre-leídos por kernel).
// Build: nvcc -O2 -arch=sm_86 -I<gpunvme>/include glm5_nvme_probe3.cu -L<build-hw> \
//   -lgpunvme_layer_loader -lgpunvme_host -lgpunvme_device -o probe3
// Uso: sudo ./probe3 <bdf> <probe3_vectors.json>
#include "gpunvme/layer_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

struct Vec {
    int layer, expert; uint64_t lba0; int in_off;
    std::vector<std::pair<uint64_t, uint32_t>> runs;
    std::vector<std::pair<uint64_t, std::string>> checks;  // pos, hex16
};

static std::string slurp(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) return "";
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, 0); size_t g = fread(&s[0], 1, n, f); (void)g; fclose(f); return s;
}
static long find_num(const std::string& s, const char* key, long from) {
    long p = (long)s.find(key, from); if (p < 0) return -1;
    return strtoull(s.c_str() + p + strlen(key), nullptr, 10);
}
static std::string find_str(const std::string& s, const char* key, long from) {
    long p = (long)s.find(key, from); if (p < 0) return "";
    const char* q = s.c_str() + p + strlen(key);
    std::string out; while (*q && *q != '"') out += *q++;
    return out;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <bdf> <vectors.json>\n", argv[0]); return 1; }
    std::string js = slurp(argv[2]);

    std::vector<Vec> vecs;
    long pos = 0;
    while ((pos = (long)js.find("\"layer\"", pos + 1)) > 0) {
        Vec v; v.layer = (int)find_num(js, "\"layer\": ", pos);
        v.expert = (int)find_num(js, "\"expert\": ", pos);
        v.lba0 = find_num(js, "\"lba0\": ", pos);
        v.in_off = (int)find_num(js, "\"in_off\": ", pos);
        // runs: "runs": [[lba,n],[lba,n]]
        long rp = (long)js.find("\"runs\": [", pos);
        long rend = (long)js.find("],", rp);
        std::string rs = js.substr(rp, rend - rp);
        for (size_t i = 0; i + 1 < rs.size(); i++) {
            if (rs[i] == '[' && (rs[i + 1] == '[' || rs[i - 1] == ',' || rs[i - 1] == ' ')) {
                uint64_t lba = strtoull(rs.c_str() + i + 1, nullptr, 10);
                size_t comma = rs.find(',', i);
                uint32_t nb = (uint32_t)strtoul(rs.c_str() + comma + 1, nullptr, 10);
                v.runs.push_back({lba, nb});
            }
        }
        // checks
        long cp = rp;
        for (;;) {
            cp = (long)js.find("{\"pos\": ", cp + 1);
            if (cp < 0 || (rend >= 0 && cp > rend + 100000)) break;
            uint64_t p2 = find_num(js, "\"pos\": ", cp);
            std::string hex = find_str(js, "\"hex\": \"", cp);
            v.checks.push_back({p2, hex});
            if (js[cp - 1] == ']') break;
        }
        vecs.push_back(v);
    }
    fprintf(stderr, "probe3: %zu vectores\n", vecs.size());

    size_t maxb = 0;
    for (auto& v : vecs) { size_t t = 0; for (auto& r : v.runs) t += r.second; if (t > maxb) maxb = t; }

    gpunvme_layer_loader_t loader;
    gpunvme_err_t e = gpunvme_layer_loader_init(&loader, argv[1], ((maxb + 4095) & ~4095ull) + 4096, 8);
    if (e != GPUNVME_OK) { fprintf(stderr, "init: %d\n", e); return 1; }

    uint8_t* buf;
    if (cudaMallocHost((void**)&buf, maxb + 8192) != cudaSuccess) { fprintf(stderr, "pin\n"); return 1; }
    uint8_t* scratch;
    if (cudaMallocHost((void**)&scratch, maxb + 8192) != cudaSuccess) { fprintf(stderr, "pin2\n"); return 1; }

    int fails = 0, checks_total = 0;
    for (size_t i = 0; i < vecs.size(); i++) {
        auto& v = vecs[i];
        memset(buf, 0, maxb + 8192);
        // fill como el engine: single-run directo; multi-run: cada run al scratch
        // (4KB-aligned) y copia con in_off, idéntico a fill_slice
        if (v.runs.size() == 1) {
            e = gpunvme_load_layer(&loader, v.runs[0].first, v.runs[0].second, buf);
            if (e != GPUNVME_OK) { printf("blk%02d e=%-3d READ FAIL %d\n", v.layer, v.expert, e); fails++; continue; }
        } else {
            uint64_t off = 0; bool bad = false;
            for (size_t k = 0; k < v.runs.size(); k++) {
                auto& r = v.runs[k];
                e = gpunvme_load_layer(&loader, r.first, r.second, scratch);
                if (e != GPUNVME_OK) { bad = true; break; }
                const uint64_t skip = (k == 0) ? (uint64_t)v.in_off : 0;
                const uint64_t take = r.second - skip;
                const uint64_t cpy = take < (maxb + 8192 - off) ? take : (maxb + 8192 - off);
                memcpy(buf + off, scratch + skip, cpy);
                off += cpy;
            }
            if (bad) { printf("blk%02d e=%-3d READ FAIL (multi)\n", v.layer, v.expert); fails++; continue; }
        }
        int local = 0;
        for (auto& c : v.checks) {
            checks_total++;
            if (memcmp(buf + v.in_off + c.first, c.second.c_str(), 16) != 0) {
                printf("blk%02d e=%-3d pos=%-8llu MISMATCH got=%.16s want=%s\n",
                       v.layer, v.expert, (unsigned long long)c.first,
                       buf + v.in_off + c.first, c.second.c_str());
                local++;
            }
        }
        if (local) fails++;
        else printf("blk%02d e=%-3d OK (%zu checks)\n", v.layer, v.expert, v.checks.size());
    }
    cudaFreeHost(buf);
    gpunvme_layer_loader_destroy(&loader);
    if (fails) printf("PROBE3 FAILED (%d vectores, %d checks)\n", fails, checks_total);
    else printf("PROBE3 OK — %d checks en %zu slices completos\n", checks_total, vecs.size());
    return fails ? 1 : 0;
}
