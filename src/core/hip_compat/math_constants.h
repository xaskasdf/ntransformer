#pragma once
// HIP build shim: resolves the project's literal `#include <math_constants.h>`
// to HIP's constants; cuda_to_hip.h maps the CUDART_* names the project uses.
#include <hip/hip_math_constants.h>
#include "../cuda_to_hip.h"
