"""Fused first-order RoPE with FP32 complex-multiply rounding and compact storage."""
import ctypes
import functools

import torch
from torch.autograd.function import once_differentiable

from .ops import checked, library


@functools.lru_cache(maxsize=1)
def _kernel():
    function = getattr(library(), 'nt_train_rotary', None)
    if function is None:
        raise RuntimeError('Fused RoPE requires rebuilding the NTransformer training library')
    function.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_int64] * 13
                         + [ctypes.c_int] * 2 + [ctypes.c_void_p])
    function.restype = ctypes.c_int
    return function


def _launch(q, k, freqs, backward):
    with torch.cuda.device(q.device):
        # Gradients from attention may be strided; read them directly instead
        # of materializing a contiguous gradient before rotating it.
        out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        checked(_kernel()(out_q.data_ptr(), out_k.data_ptr(), q.data_ptr(), k.data_ptr(),
                          freqs.data_ptr(), q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                          q.shape[3], *q.stride(), *k.stride(), int(q.dtype == torch.bfloat16),
                          int(backward), torch.cuda.current_stream().cuda_stream))
    return out_q, out_k


class _RotaryPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(freqs)
        return _launch(q, k, freqs, False)

    @staticmethod
    @once_differentiable
    def backward(ctx, dq, dk):
        freqs, = ctx.saved_tensors
        if dq is None and dk is None:
            return None, None, None
        # Preserve None for unused outputs (optimizer semantics). That rare
        # path reuses the other input to the pair kernel and discards its copy.
        q_input = (dq if dq is not None else dk).resolve_neg()
        k_input = (dk if dk is not None else dq).resolve_neg()
        q_out, k_out = _launch(q_input, k_input, freqs, True)
        return q_out if dq is not None else None, k_out if dk is not None else None, None


def reference_rotary(q, k, freqs):
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    return (torch.view_as_real(q_complex * freqs).flatten(-2).type_as(q),
            torch.view_as_real(k_complex * freqs).flatten(-2).type_as(k))


def rotary_embedding(q, k, freqs):
    supported = (q.is_cuda and q.device == k.device == freqs.device
                 and q.dtype == k.dtype and q.dtype in (torch.float32, torch.bfloat16)
                 and q.ndim == k.ndim == 4 and q.shape[:2] == k.shape[:2]
                 and q.shape[-1] == k.shape[-1] and q.shape[-1] % 2 == 0
                 and q.numel() > 0 and k.numel() > 0 and q.numel() + k.numel() < 2**31
                 and freqs.dtype == torch.complex64
                 and freqs.shape == (q.shape[1], q.shape[-1] // 2)
                 and freqs.is_contiguous() and not freqs.requires_grad
                 and not any(t.is_neg() or t.is_conj() for t in (q, k, freqs)))
    if not supported:
        return reference_rotary(q, k, freqs)
    return _RotaryPair.apply(q, k, freqs)
