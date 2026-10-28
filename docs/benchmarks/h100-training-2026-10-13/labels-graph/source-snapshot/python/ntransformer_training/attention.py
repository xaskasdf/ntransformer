"""Brandon's attention contract with compact KV heads passed directly to SDPA."""
from types import MethodType

import torch
import torch.nn.functional as F


def _direct_gqa_forward(self, x, freqs_cis, mask=None, cache=None, v_first=None):
    batch, length, _ = x.shape
    q = self.wq(x).view(batch, length, self.n_heads, self.head_dim)
    k = self.wk(x).view(batch, length, self.n_kv_heads, self.head_dim)
    v = self.wv(x).view(batch, length, self.n_kv_heads, self.head_dim)
    v_out = v
    if v_first is not None:
        v = v + v_first
    q, k = self._nt_rotary_embedding(q, k, freqs_cis)
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    new_cache = None
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
    elif not self.training:
        new_cache = (k, v)
    options = dict(dropout_p=self.dropout_p if self.training else 0.,
                   enable_gqa=self.n_heads != self.n_kv_heads)
    # Preserve the original priority: cache decode, explicit mask, then causal.
    if cache is not None:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False, **options)
    elif mask is not None:
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **options)
    else:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True, **options)
    output = output.transpose(1, 2).contiguous().view(batch, length, -1)
    return self.wo(output), new_cache, v_out


def install_direct_gqa(attention, *, fused_rope=False):
    if hasattr(attention, '_nt_rotary_embedding'):
        raise ValueError('Direct GQA already installed')
    # Keep the exact frozen Brandon RoPE implementation and frequency table.
    rotary = attention.forward.__func__.__globals__.get('apply_rotary_emb')
    if not callable(rotary) or attention.n_heads % attention.n_kv_heads:
        raise ValueError('Direct GQA requires the supported Brandon attention implementation')
    if fused_rope:
        from .rotary import rotary_embedding
        rotary = rotary_embedding
    attention._nt_rotary_embedding = rotary
    attention.forward = MethodType(_direct_gqa_forward, attention)
