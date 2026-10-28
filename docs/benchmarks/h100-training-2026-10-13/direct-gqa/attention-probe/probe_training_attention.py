#!/usr/bin/env python3
"""Replay real Brandon GQA tensors; diagnostic only, not a training acceptance gate.

Run with the same exclusive GPU admission locks as the continuation experiments.
"""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from train_brandon_experiment import import_brandon, install


def difference(a, b):
    delta = (a.float() - b.float()).double()
    return dict(equal=torch.equal(a, b), mismatches=int((a != b).sum()),
                max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm() / b.double().norm().clamp_min(1e-30)).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root', 'checkpoint', 'data-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.set_float32_matmul_precision('high')
    TinyLlama, Config, Corpus, _, _ = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = TinyLlama(Config(**checkpoint['config'])).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    install(model, shared_projection_casts=True)
    corpus = Corpus(args.data_dir, 'train', model.config.max_seq_len, checkpoint['settings']['seed'])
    batch = [t.cuda() for t in corpus.batch(16, checkpoint['step']+1, 0, 0)]
    ratio = model.config.n_heads // model.config.n_kv_heads
    captures, calls = {}, 0
    original = F.scaled_dot_product_attention

    def intercept(q, k, v, **options):
        nonlocal calls
        layer = calls
        calls += 1
        output = original(q, k, v, **options)
        if layer in (0, model.config.n_layers-1):
            compact = [q.detach().clone(), k[:, ::ratio].detach().contiguous(),
                       v[:, ::ratio].detach().contiguous()]
            assert torch.equal(compact[1].repeat_interleave(ratio, 1), k)
            assert torch.equal(compact[2].repeat_interleave(ratio, 1), v)
            captures[layer] = dict(inputs=compact, options=options)
            def upstream(gradient):
                captures[layer]['upstream'] = gradient.detach().clone()
                return gradient
            output.register_hook(upstream)
        return output

    try:
        F.scaled_dot_product_attention = intercept
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(*batch, targets_shifted=True)[1]
        loss.backward()
    finally:
        F.scaled_dot_product_attention = original
    del model, loss, checkpoint, batch
    torch.cuda.empty_cache()
    records = []
    for layer, captured in captures.items():
        path = out/f'layer-{layer}.pt'
        torch.save(captured, path)
        with path.open('rb') as handle:
            capture_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
        for backend, deterministic in ((SDPBackend.FLASH_ATTENTION, True),
                                       (SDPBackend.CUDNN_ATTENTION, False)):
            torch.use_deterministic_algorithms(deterministic)
            reference = None
            for grouped in (False, True):
                record = dict(layer=layer, backend=str(backend), deterministic=deterministic,
                              grouped=grouped, capture_sha256=capture_sha256,
                              shape=[list(t.shape) for t in captured['inputs']])
                try:
                    inputs = [x.detach().requires_grad_(True) for x in captured['inputs']]
                    def replay():
                        q, k, v = inputs
                        if not grouped:
                            k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
                        with sdpa_kernel(backend):
                            output = original(q, k, v, enable_gqa=grouped, **captured['options'])
                        grads = torch.autograd.grad(output, inputs, captured['upstream'])
                        return output, *grads
                    first = replay()
                    second = replay()
                    record['self_repeat'] = {key:difference(a, b) for key, a, b in
                                              zip(('output','dq','dk','dv'), second, first)}
                    if not grouped:
                        reference = [x.detach().clone() for x in first]
                    else:
                        record['versus_expanded'] = {key:difference(a,b) for key,a,b in
                                                    zip(('output','dq','dk','dv'), first,reference)}
                    del first, second
                    for _ in range(5):
                        replay()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    floor = torch.cuda.memory_allocated()
                    samples = []
                    for _ in range(20):
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        replay()
                        stop.record()
                        stop.synchronize()
                        samples.append(start.elapsed_time(stop))
                    record.update(median_forward_backward_ms=statistics.median(samples),
                                  samples_ms=samples, incremental_peak_bytes=torch.cuda.max_memory_allocated()-floor)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        replay()
                    record['operators'] = {row.key:row.count for row in profiler.key_averages()
                                           if any(word in row.key for word in ('scaled_dot','repeat_interleave','flash','cudnn'))}
                    record['status'] = 'measured'
                except RuntimeError as error:
                    record.update(status='unsupported', reason=str(error))
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in ('samples_ms','operators')}), flush=True)
    result = dict(status='complete', scope='Isolated real-tensor replay, not full-training throughput',
                  records=records, torch=torch.__version__, cuda=torch.version.cuda)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
