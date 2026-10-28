"""First-order CUDA autograd bridge; eager execution, FP32 norms, FP32/BF16 SwiGLU."""
import ctypes
import functools
import math
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@functools.lru_cache(maxsize=1)
def library():
    path = Path(os.environ['NTRANSFORMER_TRAINING_LIB']).resolve(strict=True)
    lib = ctypes.CDLL(str(path))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    signatures = {
        'nt_train_rms_forward': [ptr, ptr, ptr, integer, integer, ctypes.c_float, ptr],
        'nt_train_rms_backward': [ptr] * 7 + [integer, integer, ctypes.c_float, ptr],
        'nt_train_rms_apply': [ptr] * 4 + [ctypes.c_int64, integer, ptr],
        'nt_train_rms_prepare': [ptr] * 6 + [ctypes.c_int64, integer, ptr],
        'nt_train_rms_finish': [ptr] * 7 + [ctypes.c_int64, integer, ptr],
        'nt_train_swiglu_forward': [ptr] * 3 + [ctypes.c_int64, integer, ptr],
        'nt_train_swiglu_backward': [ptr] * 5 + [ctypes.c_int64, integer, ptr],
        'nt_train_enable_peer': [integer],
        'nt_train_peer_copy': [ptr, integer, ptr, integer, ctypes.c_size_t, ptr],
    }
    for name, arguments in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = integer
    lib.nt_train_error.argtypes = [integer]
    lib.nt_train_error.restype = ctypes.c_char_p
    return lib


def checked(status):
    if status:
        raise RuntimeError(library().nt_train_error(status).decode())


def validate(*tensors):
    device = tensors[0].device
    if any(t.device != device or not t.is_cuda or not t.is_contiguous() or t.numel() == 0
           for t in tensors):
        raise ValueError('Native operations require nonempty contiguous tensors on one CUDA device')


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        validate(x, weight)
        if (x.dtype != torch.float32 or weight.dtype != torch.float32 or weight.ndim != 1
                or x.ndim < 1 or x.shape[-1] != weight.numel() or not math.isfinite(eps) or eps <= 0):
            raise ValueError('RMSNorm requires FP32 input/weight, matching last dimension and positive epsilon')
        rows, dim = x.numel() // x.shape[-1], x.shape[-1]
        if x.numel() > 2**31 - 1:
            raise ValueError('RMSNorm shape exceeds kernel index range')
        with torch.cuda.device(x.device):
            y = torch.empty_like(x)
            checked(library().nt_train_rms_forward(y.data_ptr(), x.data_ptr(), weight.data_ptr(),
                    rows, dim, eps, torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.contiguous()
        rows, dim = x.numel() // x.shape[-1], x.shape[-1]
        with torch.cuda.device(x.device):
            dx, dw = torch.empty_like(x), torch.empty_like(weight)
            inverse = torch.empty(rows, device=x.device, dtype=torch.float32)
            partial = torch.empty(((rows + 127) // 128, dim), device=x.device, dtype=torch.float32)
            checked(library().nt_train_rms_backward(dx.data_ptr(), dw.data_ptr(), inverse.data_ptr(),
                    partial.data_ptr(), dy.data_ptr(), x.data_ptr(), weight.data_ptr(), rows, dim,
                    ctx.eps, torch.cuda.current_stream().cuda_stream))
        return dx, dw, None


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        validate(gate, up)
        if gate.shape != up.shape or gate.dtype != up.dtype or gate.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('SwiGLU requires equal FP32 or BF16 shapes and dtypes')
        dtype = int(gate.dtype == torch.bfloat16)
        with torch.cuda.device(gate.device):
            y = torch.empty_like(gate)
            checked(library().nt_train_swiglu_forward(y.data_ptr(), gate.data_ptr(), up.data_ptr(),
                    gate.numel(), dtype, torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(gate, up)
        ctx.dtype = dtype
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        gate, up = ctx.saved_tensors
        dy = dy.contiguous()
        with torch.cuda.device(gate.device):
            dg, du = torch.empty_like(gate), torch.empty_like(up)
            checked(library().nt_train_swiglu_backward(dg.data_ptr(), du.data_ptr(), dy.data_ptr(),
                    gate.data_ptr(), up.data_ptr(), gate.numel(), ctx.dtype,
                    torch.cuda.current_stream().cuda_stream))
        return dg, du


class _CompatibleRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, variance_x, weight, eps):
        validate(x, weight)
        if (x.dtype != torch.float32 or weight.dtype != torch.float32 or weight.ndim != 1
                or x.ndim < 1 or x.shape[-1] != weight.numel() or not math.isfinite(eps) or eps <= 0):
            raise ValueError('Compatible RMSNorm requires FP32 input/weight and positive epsilon')
        with torch.cuda.device(x.device):
            inverse = torch.rsqrt(variance_x.pow(2).mean(-1, keepdim=True) + eps)
            y = torch.empty_like(x)
            checked(library().nt_train_rms_apply(y.data_ptr(), x.data_ptr(), weight.data_ptr(),
                    inverse.data_ptr(), x.numel(), x.shape[-1], torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(x, weight, inverse)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, inverse = ctx.saved_tensors
        dy = dy.contiguous()
        with torch.cuda.device(x.device):
            row_terms, weight_terms = torch.empty_like(x), torch.empty_like(x)
            checked(library().nt_train_rms_prepare(row_terms.data_ptr(), weight_terms.data_ptr(),
                    dy.data_ptr(), x.data_ptr(), weight.data_ptr(), inverse.data_ptr(), x.numel(),
                    x.shape[-1], torch.cuda.current_stream().cuda_stream))
            row_sum = row_terms.sum(-1, keepdim=True)
            axes = tuple(range(x.ndim - 1))
            dw = weight_terms.sum(axes) if axes else weight_terms
            direct, variance = torch.empty_like(x), torch.empty_like(x)
            checked(library().nt_train_rms_finish(direct.data_ptr(), variance.data_ptr(), dy.data_ptr(), x.data_ptr(),
                    weight.data_ptr(), inverse.data_ptr(), row_sum.data_ptr(), x.numel(),
                    x.shape[-1], torch.cuda.current_stream().cuda_stream))
        return direct, variance, dw, None


rmsnorm = _RMSNorm.apply
def rmsnorm_compatible(x, weight, eps):
    # Duplicate autograd edges preserve accumulation order with external residuals.
    return _CompatibleRMSNorm.apply(x, x, weight, eps)
swiglu = _SwiGLU.apply
