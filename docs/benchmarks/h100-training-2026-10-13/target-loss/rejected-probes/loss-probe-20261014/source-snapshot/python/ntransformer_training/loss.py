"""Experimental first-order, target-only head losses; FP32 master weights stay tied.

``fused`` retains the full BF16 classifier GEMM but removes dense FP32 loss
buffers. ``vocab-stream`` recomputes BF16 logits in vocabulary blocks and sums
the resulting activation gradients in FP32 before one BF16 rounding.
"""
import ctypes
import functools
from types import MethodType

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .ops import checked, library


@functools.lru_cache(maxsize=1)
def _kernels():
    forward = getattr(library(), 'nt_train_target_logp', None)
    backward = getattr(library(), 'nt_train_target_logp_backward', None)
    if forward is None or backward is None:
        raise RuntimeError('Target-only loss requires rebuilding the training library')
    forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    backward.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    forward.restype = backward.restype = ctypes.c_int
    return forward, backward


def _forward(logits, targets, first=0, vocabulary=None):
    rows, columns = logits.shape
    with torch.cuda.device(logits.device):
        output = torch.empty(rows, device=logits.device, dtype=torch.float32)
        stats = torch.empty((3, rows), device=logits.device, dtype=torch.float32)
        checked(_kernels()[0](output.data_ptr(), stats.data_ptr(), logits.data_ptr(),
                             targets.data_ptr(), rows, columns, first,
                             vocabulary or columns, torch.cuda.current_stream().cuda_stream))
    return output, stats


def _backward(logits, targets, stats, gradient, first=0):
    with torch.cuda.device(logits.device):
        dz = torch.empty_like(logits)
        checked(_kernels()[1](dz.data_ptr(), logits.data_ptr(), stats.data_ptr(),
                             targets.data_ptr(), gradient.data_ptr(), *logits.shape,
                             first, torch.cuda.current_stream().cuda_stream))
    return dz


class _TargetLogProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        output, stats = _forward(logits, targets)
        ctx.save_for_backward(logits, targets, stats)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        logits, targets, stats = ctx.saved_tensors
        return _backward(logits, targets, stats, gradient.contiguous()), None


class _VocabularyStream(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, block_size):
        # Autocast is already made explicit on the two inputs. In particular,
        # backward's FP32 GEMM output must not be autocast back to BF16.
        with torch.autocast('cuda', enabled=False):
            stats = None
            for start in range(0, weight.shape[0], block_size):
                logits = torch.mm(h, weight[start:start + block_size].t())
                _, current = _forward(logits, targets, start, weight.shape[0])
                if stats is None:
                    stats = current
                else:
                    maximum = torch.maximum(stats[0], current[0])
                    total = (stats[1] * torch.exp(stats[0] - maximum)
                             + current[1] * torch.exp(current[0] - maximum))
                    stats = torch.stack((maximum, total, stats[2] + current[2]))
                del logits
            output = (stats[2] - stats[0]) - torch.log(stats[1])
        ctx.save_for_backward(h, weight, targets, stats)
        ctx.block_size = block_size
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        h, weight, targets, stats = ctx.saved_tensors
        gradient = gradient.contiguous()
        with torch.autocast('cuda', enabled=False):
            dh = None
            dw = torch.empty_like(weight) if ctx.needs_input_grad[1] else None
            for start in range(0, weight.shape[0], ctx.block_size):
                block = weight[start:start + ctx.block_size]
                logits = torch.mm(h, block.t())
                dz = _backward(logits, targets, stats, gradient, start)
                if ctx.needs_input_grad[0]:
                    partial = torch.mm(dz, block, out_dtype=torch.float32)
                    if dh is None:
                        dh = partial
                    else:
                        dh.add_(partial)
                if dw is not None:
                    torch.mm(dz.t(), h, out=dw[start:start + ctx.block_size])
                del logits, dz
        return dh.to(h.dtype) if dh is not None else None, dw, None, None


def linear_cross_entropy(h, weight, targets, target_mask=None, *, targets_shifted=True,
                         mode='fused', block_size=2048):
    """Loss-only counterpart of Brandon's default CE, including shifts and masks.

    Narrow native support: CUDA BF16 compute, 8192 vocabulary, integer labels,
    no smoothing or auxiliary losses. Other compute shapes use ordinary CE.
    """
    if mode not in ('fused', 'vocab-stream'):
        raise ValueError('Unsupported target-only loss mode')
    if not isinstance(block_size, int) or not 1 <= block_size <= 8192:
        raise ValueError('Vocabulary block size must be in 1..8192')
    if h.ndim != 3 or targets.shape != h.shape[:2] or weight.ndim != 2 or weight.shape[1] != h.shape[2]:
        raise ValueError('Expected hidden [B,T,D], labels [B,T], weight [V,D]')
    if target_mask is not None and target_mask.shape != targets.shape:
        raise ValueError('Target mask shape must match labels')
    offset = 0 if targets_shifted else 1
    end = h.shape[1] - offset
    labels = targets[..., offset:].contiguous().view(-1)
    supported = (h.is_cuda and h.device == weight.device == targets.device
                 and targets.dtype == torch.int64 and weight.shape[0] == 8192
                 and h.numel() > 0 and end > 0
                 and ((torch.is_autocast_enabled('cuda')
                       and torch.get_autocast_dtype('cuda') == torch.bfloat16)
                      or h.dtype == weight.dtype == torch.bfloat16))
    if not supported:
        logits = F.linear(h, weight)[..., :end, :].contiguous().view(-1, weight.shape[0])
        losses = F.cross_entropy(logits, labels, reduction='none' if target_mask is not None else 'mean')
    else:
        # Keep all B*T rows for classifier GEMMs, even for shifted SFT. This
        # preserves the reference's zero-gradient final rows and dW shape.
        all_labels = (targets if offset == 0 else
                      F.pad(targets[..., 1:], (0, 1), value=-100)).contiguous().view(-1)
        torch._assert_async(((all_labels == -100) | ((all_labels >= 0) &
                            (all_labels < weight.shape[0]))).all(), 'Target index out of bounds')
        with torch.autocast('cuda', enabled=False):
            hidden = h.to(torch.bfloat16).reshape(-1, h.shape[-1])
            w = weight.to(torch.bfloat16)
            if mode == 'fused':
                logp = _TargetLogProbability.apply(torch.mm(hidden, w.t()), all_labels)
            else:
                logp = _VocabularyStream.apply(hidden, w, all_labels, block_size)
            logp = logp.view(*targets.shape)[..., :end].contiguous().view(-1, 1)
            # The same NLL row reduction supplies both the scalar and its
            # per-row gradient, retaining ignore_index and mean semantics.
            proxy = torch.where(labels == -100, -100, 0)
            losses = F.nll_loss(logp, proxy, reduction='none' if target_mask is not None else 'mean')
    if target_mask is not None:
        mask = target_mask[..., offset:].contiguous().view(-1)
        return (losses * mask).sum() / (mask.sum() + 1e-8)
    return losses


def _loss_forward(self, tokens, targets=None, target_mask=None, n_predict_override=None,
                  label_smoothing=0., unlikelihood_alpha=0., entropy_reg_beta=0.,
                  targets_shifted=False, *, loss_only=False):
    if (not loss_only or targets is None or label_smoothing != 0 or unlikelihood_alpha != 0
            or entropy_reg_beta != 0 or n_predict_override not in (None, 1)):
        return self._nt_original_forward(tokens, targets, target_mask, n_predict_override,
                                         label_smoothing, unlikelihood_alpha,
                                         entropy_reg_beta, targets_shifted)
    if tokens.shape[1] > self.config.max_seq_len:
        raise ValueError('Sequence exceeds model context')
    h = self.dropout(self.tok_embeddings(tokens))
    h, _ = self._run_layers(h, self.freqs_cis[:tokens.shape[1]], v_first_in=None)
    loss = linear_cross_entropy(self.norm(h), self.output.weight, targets, target_mask,
                                targets_shifted=targets_shifted, mode=self._nt_loss_mode,
                                block_size=self._nt_loss_block_size)
    return None, loss


def install_loss(model, mode, block_size):
    if mode not in ('fused', 'vocab-stream') or not 1 <= block_size <= 8192:
        raise ValueError('Invalid target-only loss configuration')
    model._nt_original_forward = model.forward
    model._nt_loss_mode, model._nt_loss_block_size = mode, block_size
    model.forward = MethodType(_loss_forward, model)
