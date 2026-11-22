// Stand-ins for the DS41 / GLM5 CLI entry points in builds that leave the MoE
// engines out (USE_HIP: the HIP port covers the Llama path only).
//
// Like the real run_cli, each one declines (-1) for any other architecture so
// main.cpp falls through to the Llama engine. For its own architecture it
// reports that the build lacks the engine instead of letting the Llama loader
// fail on an unfamiliar tensor layout.

#include "model/loader.h"
#include <cstdio>
#include <string>

namespace nt {
namespace glm5 { struct GenOptions; int run_cli(const std::string&, const GenOptions&); }
namespace ds41 { struct GenOptions; int run_cli(const std::string&, const GenOptions&); }

namespace {
int reject_arch(const std::string& model_path, const char* arch, const char* name) {
    GGUFLoader peek;
    if (!peek.load(model_path)) return -1;
    if (peek.kv_string("general.architecture", "") != arch) return -1;
    fprintf(stderr, "%s (%s) is not available in this build: the MoE engines are "
            "CUDA-only (rebuild without USE_HIP)\n", name, arch);
    return 1;
}
} // namespace

int glm5::run_cli(const std::string& model_path, const GenOptions&) {
    return reject_arch(model_path, "glm5-next", "GLM-5.3-Flash");
}

int ds41::run_cli(const std::string& model_path, const GenOptions&) {
    return reject_arch(model_path, "deepseek41", "DeepSeek-V4.1-Flash");
}
} // namespace nt
