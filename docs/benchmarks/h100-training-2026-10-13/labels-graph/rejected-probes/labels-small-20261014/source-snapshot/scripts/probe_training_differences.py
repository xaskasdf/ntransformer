#!/usr/bin/env python3
"""Isolate first-pretraining-batch gradients and SDPA repeatability on CUDA."""
import argparse
import copy
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_brandon_experiment import import_brandon, install


def differences(actual, expected):
    records = []
    for name, a in actual.items():
        b = expected[name]
        delta = a.double() - b.double()
        records.append(dict(name=name, mismatches=int((a != b).sum()), elements=a.numel(),
                            max_abs=delta.abs().max().item(), error_l2_squared=delta.square().sum().item(),
                            reference_l2_squared=b.double().square().sum().item()))
    error = sum(r['error_l2_squared'] for r in records)
    magnitude = sum(r['reference_l2_squared'] for r in records)
    return dict(differing_tensors=sum(r['mismatches'] > 0 for r in records), total_tensors=len(records),
                mismatches=sum(r['mismatches'] for r in records),
                relative_l2=(error / max(magnitude, 1e-300)) ** .5,
                max_abs=max(r['max_abs'] for r in records),
                tensors=sorted(records, key=lambda r: r['error_l2_squared'], reverse=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--objective', choices=['pretrain', 'sft'], default='pretrain')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2026)
    TinyLlama, ModelConfig, PackedCorpus, _, trainer = import_brandon(args.brandon_root)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    seed = TinyLlama(ModelConfig(**payload['config']))
    seed.load_state_dict(payload['model_state_dict'])
    if args.objective == 'pretrain':
        corpus = PackedCorpus(args.data_dir, 'train', seed.config.max_seq_len, payload['settings']['seed'])
        batch = corpus.batch(payload['settings']['batch_size'], payload['step'], 0, 0)
    else:
        rows = trainer.read_instruction_jsonl(Path(args.data_dir) / 'train.jsonl')
        tokenizer = trainer.Tokenizer(str(Path(args.brandon_root) / 'data/tokenizer_8k.model'))
        rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, 0, 0]))
        batch = trainer.instruction_batch([rows[i] for i in rng.integers(len(rows), size=8)], tokenizer, 1024)
    batch_sha = hashlib.sha256(b''.join(t.numpy().tobytes() for t in batch)).hexdigest()
    batch = [t.cuda() for t in batch]
    result = dict(torch=torch.__version__, torch_git=torch.version.git_version, objective=args.objective,
                  batch_sha256=batch_sha, shape=list(batch[0].shape), model_runs=[], attention_runs=[])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def save():
        destination.write_text(json.dumps(result, indent=2) + '\n')
    original_attention = F.scaled_dot_product_attention
    captured = None
    def attention(*values, **options):
        nonlocal captured
        if captured is None:
            captured = ([v.detach().clone() for v in values[:3]], dict(options))
        return original_attention(*values, **options)
    F.scaled_dot_product_attention = attention
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        reference = None
        for mode in ('reference', 'reference-repeat', 'norms', 'swiglu', 'native', 'checkpoint'):
            record = dict(deterministic=deterministic, mode=mode)
            try:
                model = copy.deepcopy(seed).cuda()
                if mode not in ('reference', 'reference-repeat'):
                    install(model, native_norms=mode != 'swiglu', native_swiglu=mode != 'norms',
                            norm_mode='compatible', checkpoint=mode == 'checkpoint')
                torch.manual_seed(2026)
                profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA])
                with profile if mode == 'reference' else nullcontext():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(*batch, targets_shifted=args.objective == 'pretrain')[1]
                    loss.backward()
                gradients = {name: p.grad.cpu().clone() for name, p in model.named_parameters()}
                record['loss'] = loss.item()
                if mode == 'reference':
                    reference = gradients
                    record['attention_operators'] = [e.key for e in profile.key_averages()
                                                     if 'attention' in e.key.lower()]
                else:
                    record['gradient_difference'] = differences(gradients, reference)
                del model, gradients, loss
                torch.cuda.empty_cache()
            except RuntimeError as error:
                record['error'] = str(error)
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
            result['model_runs'].append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'gradient_difference'} |
                             {'gradient_difference': {k:v for k,v in record.get('gradient_difference',{}).items()
                                                      if k != 'tensors'}}), flush=True)
            save()
            if mode == 'reference' and reference is None:
                break
    F.scaled_dot_product_attention = original_attention
    values, options = captured
    torch.manual_seed(1234)
    upstream = torch.randn_like(values[0])
    for backend in ('auto', 'flash'):
        for deterministic in (False, True):
            torch.use_deterministic_algorithms(deterministic)
            reference = None
            for repeat in range(3):
                record = dict(backend=backend, deterministic=deterministic, repeat=repeat,
                              input_shapes=[list(v.shape) for v in values],
                              input_dtypes=[str(v.dtype) for v in values], autocast='bfloat16')
                try:
                    inputs = [v.detach().requires_grad_() for v in values]
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend == 'flash' else nullcontext():
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            output = original_attention(*inputs, **options)
                        gradients = torch.autograd.grad(output, inputs, upstream)
                    actual = {name:g.detach().cpu() for name,g in zip(('q', 'k', 'v'), gradients)}
                    if reference is None:
                        reference = actual
                    else:
                        record['gradient_difference'] = differences(actual, reference)
                except RuntimeError as error:
                    record['error'] = str(error)
                result['attention_runs'].append(record)
                save()
    print('Probe complete', flush=True)


if __name__ == '__main__':
    main()
