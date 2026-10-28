"""Share forward casts without merging the FP32 backward accumulation edges.

Scoped to the sequential, read-only projection groups in the Brandon adapter.
Weights and GEMMs still use PyTorch autocast. No tensors survive in the scope
after forward; autograd keeps the shared BF16 storage as long as GEMMs need it.
"""
from types import MethodType

import torch
from torch.autograd.function import once_differentiable


class _BranchCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, converted):
        ctx.source_dtype = source.dtype
        return converted

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Each branch converts before autograd sums at the original FP32 input.
        # A single differentiable shared .to(BF16) would sum in BF16 first.
        return gradient.to(ctx.source_dtype), None


class _CastScope:
    def __init__(self):
        self.current = None


def _group_forward(self, *args, **kwargs):
    scope = self._nt_cast_scope
    previous = scope.current
    scope.current = []
    try:
        return self._nt_original_projection_forward(*args, **kwargs)
    finally:
        scope.current = previous


def _linear_forward(self, input):
    cache = self._nt_cast_scope.current
    # Autocast may already cache trainable leaves with a shared backward edge.
    # Preserve that behavior; Brandon norm outputs are nonleaf activations.
    if (cache is None or not input.is_cuda or input.dtype != torch.float32
            or (input.is_leaf and input.requires_grad)
            or input.is_inference() or not torch.is_autocast_enabled('cuda')
            or torch.get_autocast_dtype('cuda') != torch.bfloat16):
        return self._nt_original_projection_forward(input)
    if not cache or cache[0] is not input or cache[1] != input._version:
        with torch.no_grad():
            converted = input.to(torch.bfloat16)
        cache[:] = [input, input._version, converted]
    converted = _BranchCast.apply(input, cache[2]) if torch.is_grad_enabled() else cache[2]
    return self._nt_original_projection_forward(converted)


def share_projection_casts(module, names):
    """Wrap a projection group; preserve its forward, module keys and parameters.

The supported trainer runs one forward per module at a time on its host thread.
Nested sequential calls restore their enclosing scope. Concurrent host calls to
the same module and higher-order gradients are outside the adapter's contract.
"""
    if hasattr(module, '_nt_cast_scope'):
        raise ValueError('Shared projection casts already installed')
    projections = [getattr(module, name) for name in names]
    if len(projections) < 2 or any(type(layer) is not torch.nn.Linear for layer in projections):
        raise ValueError('Shared casts require at least two dense Linear projections')
    scope = _CastScope()
    module._nt_cast_scope = scope
    module._nt_original_projection_forward = module.forward
    module.forward = MethodType(_group_forward, module)
    for layer in projections:
        layer._nt_cast_scope = scope
        layer._nt_original_projection_forward = layer.forward
        layer.forward = MethodType(_linear_forward, layer)
