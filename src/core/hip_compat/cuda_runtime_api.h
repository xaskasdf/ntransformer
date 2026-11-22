#pragma once
// HIP build shim: resolves the project's literal `#include <cuda_runtime_api.h>`
// to the CUDA->HIP aliases. See cuda_runtime.h in this directory.
#include "../cuda_to_hip.h"
