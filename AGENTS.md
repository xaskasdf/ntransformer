# Repository Guidelines

## Project Structure & Module Organization

NTransformer is a C++/CUDA Llama inference engine with GPU layer streaming.
- `src/core/`: tensors, allocators, and device management.
- `src/cuda/`: kernels; `src/memory/`: layer streaming and caching.
- `src/model/`: transformer components and GGUF loading; `src/inference/`: tokenization, sampling, and execution.
- `src/utils/`: logging, profiling, and timing; `src/main.cpp`: CLI entry point.
- `include/ntransformer.h`: public C API header; `tests/`: standalone tests.
- `scripts/`: Linux/NVMe setup; `tools/`: GGUF decomposition; `docs/`: architecture, development, and research notes.

Keep model assets external; `.gguf`, `.bin`, and `.safetensors` files are ignored.

## Build, Test, and Development Commands

Use Linux, CMake 3.24+, gcc/g++-14, CUDA Toolkit 13.1, and an NVIDIA GPU with compute capability 8.0+. Run from the repository root:

```bash
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=gcc-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build -j
# Run registered tests with visible diagnostics
ctest --test-dir build -V
# Generate text from a local model
./build/ntransformer -m /path/to/model.gguf -p "Hello" -n 128
```

Add `--streaming` for models exceeding VRAM, or `--benchmark -n 64` for throughput measurements. Optional `-DUSE_GPUNVME=ON` requires prebuilt libraries in `../gpu-nvme-direct/build-hw/`.

## Coding Style & Naming Conventions

Use C++20, four-space indentation, and same-line opening braces. Follow existing `PascalCase` types, `snake_case` functions/files, and trailing underscores for private members. Use `nt` and `nt::cuda` namespaces, `launch_*` kernel wrappers, and `nt_cuda_*` C-linkage device helpers. No formatter or linter configuration is checked in; match surrounding code.

## Testing Guidelines

Tests use standalone executables, assertions, and numerical comparisons; no external framework or coverage threshold is configured. Name files `tests/test_<feature>.cpp` or `.cu` and register automated tests in CMake. Reconfigure with `-DCMAKE_BUILD_TYPE=Debug` to retain assertions. Inspect verbose output for `FAIL` and `SKIP`: GPU checks may skip, and some failures do not affect exit status. `test_nvme_layer` is hardware-dependent and is not registered with CTest.

## Commit & Pull Request Guidelines

History uses descriptive subjects, often prefixed by subsystem, such as `Async NVMe prefetch: ...`; Conventional Commits are not required. Keep commits focused. PRs should explain behavior changes, link relevant issues, list validation commands and hardware, and include before/after throughput and VRAM measurements for performance changes.

## Hardware Configuration

Read `docs/GPU_NVME_DIRECT_INTEGRATION.md` before NVMe work. Setup scripts modify drivers and device bindings; use a dedicated secondary NVMe, never the boot drive.
