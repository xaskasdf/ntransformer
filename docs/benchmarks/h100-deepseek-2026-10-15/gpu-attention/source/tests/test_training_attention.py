#!/usr/bin/env python3
"""Direct GQA: full-model gradients plus mask, KV-cache and decode semantics."""
import argparse
import copy
import json
from pathlib import Path

from test_training_projection_casts import model_checks, import_brandon, equal, torch
from ntransformer_training.attention import install_direct_gqa


def attention_checks(args, *, fused_rope=False):
    TinyLlama, Config, _, _, _ = import_brandon(args.brandon_root)
    config = Config(dim=576, n_layers=1, n_heads=9, n_kv_heads=3,
                    hidden_dim=1472, vocab_size=64, max_seq_len=32, dropout=0.)
    seed = TinyLlama(config).cuda()
    reference = seed.layers[0].attention
    candidate = copy.deepcopy(reference)
    install_direct_gqa(candidate, fused_rope=fused_rope)
    freqs = seed.freqs_cis
    records = []
    for name in ('causal', 'mask', 'prefill', 'decode'):
        reference.train(name in ('causal', 'mask'))
        candidate.train(reference.training)
        length = 1 if name == 'decode' else 17
        x = torch.randn(2, length, 576, device='cuda', requires_grad=True)
        mask = torch.ones(length, length, dtype=torch.bool, device='cuda').tril() if name == 'mask' else None
        cache = tuple(torch.randn(2, 3, 8, 64, device='cuda', dtype=torch.bfloat16)
                      for _ in range(2)) if name == 'decode' else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = reference(x, freqs[:length], mask=mask, cache=cache)
            actual = candidate(x, freqs[:length], mask=mask, cache=cache)
        equal(actual[0], expected[0])
        equal(actual[2], expected[2])
        if expected[1] is not None:
            for a, b in zip(actual[1], expected[1]):
                equal(a, b)
            assert actual[1][0].shape[1] == 3
        else:
            assert actual[1] is None
        probe = torch.randn_like(expected[0]) * 1e-4
        ga = torch.autograd.grad(expected[0], (x, *reference.parameters()), probe)
        gb = torch.autograd.grad(actual[0], (x, *candidate.parameters()), probe)
        for a, b in zip(ga, gb):
            equal(a, b)
        records.append(dict(case=name, exact=True))
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
    result = dict(attention=attention_checks(args), model=model_checks(args, direct_gqa=True), status='passed')
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
