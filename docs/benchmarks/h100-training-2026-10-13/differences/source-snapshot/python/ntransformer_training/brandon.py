"""Explicit adapter for the validated dense Brandon Mini architecture."""
from types import MethodType

import torch

from .ops import rmsnorm, rmsnorm_compatible, swiglu
from .pipeline import checkpoint_block


def validate_config(config):
    disabled = ('block_sharing', 'dense_former', 'value_residual', 'n_registers',
                'ternary', 'relu_squared', 'step_embedding', 'feedback')
    if any(getattr(config, name, False) for name in disabled):
        raise ValueError('This experiment supports the dense, unshared Brandon Mini architecture only')
    if config.n_loops != 1 or config.n_predict != 1 or config.dropout != 0:
        raise ValueError('The checkpoint pipeline currently requires one loop/head and zero dropout')


def _norm_forward(self, x):
    return rmsnorm(x.contiguous(), self.weight, self.eps)


def _compatible_norm_forward(self, x):
    return rmsnorm_compatible(x.contiguous(), self.weight, self.eps)


def _ffn_forward(self, x):
    return self.dropout(self.w2(swiglu(self.w1(x), self.w3(x))))


def _checkpoint_layers(self, h, freqs_cis, v_first_in=None):
    if v_first_in is not None:
        raise ValueError('Value residual is unsupported by the training pipeline')
    if not self.training or not torch.is_grad_enabled():
        return self._nt_original_run_layers(h, freqs_cis, v_first_in)
    pool = self._nt_peer_pool
    if pool is not None:
        pool.begin()
    for index, layer in enumerate(self.layers):
        h = checkpoint_block(h, layer, freqs_cis, pool, index)
    return h, None


def install(model, *, native=True, checkpoint=False, peer_pool=None,
            native_norms=True, native_swiglu=True, norm_mode='compatible'):
    validate_config(model.config)
    if norm_mode not in ('fused', 'compatible'):
        raise ValueError('Unsupported RMSNorm implementation')
    if getattr(model, '_nt_installed', False):
        raise ValueError('Training backend already installed')
    if peer_pool is not None and not checkpoint:
        raise ValueError('Peer activation storage requires block checkpointing')
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError('Training requires all parameters trainable in FP32 (BF16 autocast for compute)')
    if native and native_norms:
        forward = _compatible_norm_forward if norm_mode == 'compatible' else _norm_forward
        model.norm.forward = MethodType(forward, model.norm)
        for layer in model.layers:
            for norm in (layer.attention_norm, layer.ffn_norm):
                norm.forward = MethodType(forward, norm)
    if native and native_swiglu:
        for layer in model.layers:
            layer.feed_forward.forward = MethodType(_ffn_forward, layer.feed_forward)
    if checkpoint:
        model._nt_original_run_layers = model._run_layers
        model._nt_peer_pool = peer_pool
        model._run_layers = MethodType(_checkpoint_layers, model)
    model._nt_installed = True
    return dict(native_norms=2 * len(model.layers) + 1 if native and native_norms else 0,
                norm_mode=norm_mode if native and native_norms else 'reference',
                norm_reductions='native CUDA' if native and native_norms and norm_mode == 'fused' else 'PyTorch',
                native_swiglu=len(model.layers) if native and native_swiglu else 0,
                checkpoint_layers=len(model.layers) if checkpoint else 0,
                peer_activation_storage=peer_pool is not None,
                gemm_attention_optimizer_ddp='PyTorch', higher_order_gradients=False)
