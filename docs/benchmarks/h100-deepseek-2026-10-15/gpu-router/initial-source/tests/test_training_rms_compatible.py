#!/usr/bin/env python3
"""Exact RMSNorm rounding checks, including the real masked SFT first batch."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_brandon_experiment import import_brandon, install
from ntransformer_training.ops import rmsnorm_compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(4141)
    records = []
    for shape in [(7,), (17, 1), (3, 33), (2, 7, 576), (257, 576), (19, 4097)]:
        x = torch.randn(shape, device='cuda', requires_grad=True)
        w = torch.randn(shape[-1], device='cuda', requires_grad=True)
        dy = torch.randn_like(x) * 1e-5
        expected = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)) * w
        actual = rmsnorm_compatible(x, w, 1e-5)
        a = torch.autograd.grad(actual, (x, w), dy)
        b = torch.autograd.grad(expected, (x, w), dy)
        record = dict(shape=shape, forward=(actual != expected).sum().item(),
                      dx=(a[0] != b[0]).sum().item(), dw=(a[1] != b[1]).sum().item())
        records.append(record)
        print(json.dumps(record), flush=True)
        for left, right in [(actual, expected), *zip(a, b)]:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    TinyLlama, ModelConfig, _, _, trainer = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    seed = TinyLlama(ModelConfig(**checkpoint['config']))
    seed.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    data = trainer.read_instruction_jsonl(Path(args.data_dir) / 'train.jsonl')
    tokenizer = trainer.Tokenizer(str(Path(args.brandon_root) / 'data/tokenizer_8k.model'))
    rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, 0, 0]))
    batch = [tensor.cuda() for tensor in trainer.instruction_batch(
        [data[i] for i in rng.integers(len(data), size=8)], tokenizer, 1024)]
    reference = None
    for mode in ('reference', 'compatible'):
        model = copy.deepcopy(seed).cuda()
        if mode == 'compatible':
            install(model, norm_mode='compatible')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(*batch, targets_shifted=False)[1]
        loss.backward()
        gradients = {name: p.grad.cpu().clone() for name, p in model.named_parameters()}
        if reference is None:
            reference, expected_loss = gradients, loss.item()
        else:
            assert loss.item() == expected_loss
            errors = [(name, (grad != reference[name]).sum().item(),
                       (grad - reference[name]).abs().max().item()) for name, grad in gradients.items()
                      if not torch.equal(grad, reference[name])]
            print(json.dumps(dict(differing_parameters=len(errors), examples=errors[:8])), flush=True)
            for name, grad in gradients.items():
                torch.testing.assert_close(grad, reference[name], rtol=0, atol=0, msg=name)
        del model
    result = dict(status='passed', primitives=records, real_sft=dict(loss=expected_loss,
                  exact_gradients=len(reference), parameters=seed.count_parameters()))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
