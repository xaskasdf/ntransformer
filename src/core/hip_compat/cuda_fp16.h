#pragma once
// HIP build shim: resolves the project's literal `#include <cuda_fp16.h>` to
// HIP's half type. See cuda_runtime.h in this directory for the rationale.
#include "../cuda_to_hip.h"
