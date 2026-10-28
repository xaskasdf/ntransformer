#!/usr/bin/env python3
"""Exact RoPE rounding, strided gradients, fallback and full Brandon checks."""
import argparse
import json
from pathlib import Path

from test_training_attention import attention_checks
from test_training_projection_casts import model_checks, equal, torch
from ntransformer_training.rotary import rotary_embedding, reference_rotary


def primitive_checks():
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for length, dim, scale, layout in ((17, 64, 1., 'dense'),
                                          (19, 6, 1e-5, 'strided'),
                                          (1, 2, 1e-10, 'broadcast_gradient'),
                                          (1024, 64, 1e-5, 'dense')):
            q = torch.randn(2, 9, length, dim, device='cuda', dtype=dtype).transpose(1, 2)
            k = torch.randn(2, 3, length, dim, device='cuda', dtype=dtype).transpose(1, 2)
            if layout != 'strided':
                q, k = q.contiguous(), k.contiguous()
            q, k = q.requires_grad_(), k.requires_grad_()
            # Slice with a position offset; no frequency regeneration in the kernel.
            angles = torch.randn(length+7, dim//2, device='cuda') * 100
            freqs = torch.polar(torch.ones_like(angles), angles)[7:]
            expected = reference_rotary(q, k, freqs)
            actual = rotary_embedding(q, k, freqs)
            assert type(actual[0].grad_fn).__name__ == '_RotaryPairBackward'
            saved = actual[0].grad_fn.saved_tensors
            assert len(saved) == 1 and saved[0].data_ptr() == freqs.data_ptr()
            for a, b in zip(actual, expected):
                equal(a, b)
            upstream = []
            for x in actual:
                if layout == 'broadcast_gradient':
                    g = (torch.randn(1, device='cuda', dtype=dtype)*scale).expand(x.shape)
                else:
                    g = (torch.randn(*x.shape[:-1], dim*2, device='cuda', dtype=dtype)*scale)[..., ::2]
                upstream.append(g)
            a = torch.autograd.grad(actual, (q, k), upstream, retain_graph=True)
            b = torch.autograd.grad(expected, (q, k), upstream, retain_graph=True)
            for left, right in zip(a, b):
                equal(left, right)
            for index in (0, 1):
                a = torch.autograd.grad(actual[index], (q, k), upstream[index],
                                        allow_unused=True, retain_graph=True)
                b = torch.autograd.grad(expected[index], (q, k), upstream[index],
                                        allow_unused=True, retain_graph=True)
                assert a[1-index] is b[1-index] is None
                equal(a[index], b[index])
            records.append(dict(dtype=str(dtype),length=length,dim=dim,gradient_scale=scale,
                                layout=layout,exact=True,saved_inputs=False))
    for case in ('cpu', 'trainable_frequency', 'conjugate_frequency', 'fp64', 'mixed_dtype'):
        device = 'cpu' if case == 'cpu' else 'cuda'
        dtype = torch.float64 if case == 'fp64' else torch.float32
        q = torch.randn(2, 7, 9, 6, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(2, 7, 3, 6, device=device,
                        dtype=torch.bfloat16 if case == 'mixed_dtype' else dtype, requires_grad=True)
        angles = torch.randn(7, 3, device=device)
        freqs = torch.polar(torch.ones_like(angles), angles)
        if case == 'conjugate_frequency':
            freqs = freqs.conj()
        if case == 'trainable_frequency':
            freqs.requires_grad_()
        expected, actual = reference_rotary(q, k, freqs), rotary_embedding(q, k, freqs)
        assert type(actual[0].grad_fn).__name__ != '_RotaryPairBackward'
        for a, b in zip(actual, expected):
            equal(a, b)
        inputs = (q, k, freqs) if freqs.requires_grad else (q, k)
        probes = [torch.randn_like(x)*1e-5 for x in actual]
        for a, b in zip(torch.autograd.grad(actual, inputs, probes),
                        torch.autograd.grad(expected, inputs, probes)):
            equal(a, b)
        records.append(dict(fallback=case,exact=True))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root', 'checkpoint', 'data-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(20261014)
    result = dict(primitives=primitive_checks())
    print('RoPE primitives and fallbacks exact', flush=True)
    result['attention'] = attention_checks(args, fused_rope=True)
    result['model'] = model_checks(args, direct_gqa=True, fused_rope=True)
    result['status'] = 'passed'
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
