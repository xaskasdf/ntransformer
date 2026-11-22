#pragma once
// HIP build shim: this directory is placed on the include path only for the
// ROCm/HIP build so the project's literal `#include <cuda_runtime.h>` resolves
// to the CUDA->HIP compatibility aliases. The NVIDIA build never sees this dir
// and picks up the real Toolkit header.
#include "../cuda_to_hip.h"
