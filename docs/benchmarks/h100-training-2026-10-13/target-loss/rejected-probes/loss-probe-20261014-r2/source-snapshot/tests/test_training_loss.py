#!/usr/bin/env python3
"""Target-only loss: labels/masks/shifts, tied gradients, full Brandon batches."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from ntransformer_training.loss import _TargetLogProbability, linear_cross_entropy
from train_brandon_experiment import import_brandon, install


def difference(actual, expected):
    a, b = actual.double(), expected.double()
    return dict(exact=torch.equal(actual, expected), max_abs=(a-b).abs().max().item(),
                relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item())


def primitives():
    records = []
    for scale in (1., 6., 30.):
        z = (torch.randn(257, 8192, device='cuda') * scale).bfloat16().requires_grad_()
        labels = torch.randint(8192, (257,), device='cuda')
        reference = F.log_softmax(z.float(), dim=-1).gather(1, labels[:, None]).squeeze(1)
        actual = _TargetLogProbability.apply(z, labels)
        probe = torch.randn_like(actual) * 1e-4
        ga, = torch.autograd.grad(actual, z, probe)
        gb, = torch.autograd.grad(reference, z, probe)
        record = dict(scale=scale, logp=difference(actual, reference), gradient=difference(ga, gb))
        print(json.dumps(record), flush=True)
        records.append(record)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    return records


def semantics():
    records = []
    for shifted, mask_kind in ((True, 'none'), (False, 'none'), (False, 'weighted'),
                               (True, 'zero'), (True, 'ignored'), (False, 'ignored-mask')):
        h0 = torch.randn(2, 33, 64, device='cuda')
        w0 = torch.randn(8192, 64, device='cuda') * .1
        labels = torch.randint(8192, (2, 33), device='cuda')
        labels[:, :7] = -100
        mask = None
        if mask_kind == 'weighted':
            mask = torch.rand(2, 33, device='cuda')
        elif mask_kind == 'zero':
            mask = torch.zeros(2, 33, device='cuda')
        if mask_kind.startswith('ignored'):
            labels.fill_(-100)
            if mask_kind.endswith('mask'):
                mask = torch.ones_like(labels, dtype=torch.float32)
        expected = None
        for mode in ('reference', 'fused', 'vocab-stream'):
            h, w = h0.detach().requires_grad_(), w0.detach().requires_grad_()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if mode == 'reference':
                    offset = 0 if shifted else 1
                    z = F.linear(h, w)[..., :33-offset, :].contiguous().view(-1, 8192)
                    loss = F.cross_entropy(z, labels[..., offset:].contiguous().view(-1),
                                           reduction='mean' if mask is None else 'none')
                    if mask is not None:
                        m = mask[..., offset:].contiguous().view(-1)
                        loss = (loss*m).sum()/(m.sum()+1e-8)
                else:
                    loss = linear_cross_entropy(h, w, labels, mask, targets_shifted=shifted, mode=mode)
            gradients = torch.autograd.grad(loss, (h, w))
            if expected is None:
                expected = (loss, gradients)
                continue
            print(json.dumps(dict(mode=mode, shifted=shifted, mask=mask_kind,
                                  loss=loss.item(), reference_loss=expected[0].item())), flush=True)
            torch.testing.assert_close(loss, expected[0], rtol=0, atol=2e-6, equal_nan=True)
            metrics = [difference(a, b) for a, b in zip(gradients, expected[1])]
            if mode == 'fused':
                for a, b in zip(gradients, expected[1]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                assert all(m['relative_l2'] < .01 for m in metrics), metrics
            records.append(dict(mode=mode, shifted=shifted, mask=mask_kind, gradients=metrics))
    # CPU fallback keeps higher precision and ordinary CE semantics.
    h = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    y = torch.randint(11, (2, 4))
    a = linear_cross_entropy(h, w, y)
    b = F.cross_entropy(F.linear(h, w).view(-1, 11), y.view(-1))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    return records


def model_checks(args):
    TinyLlama, ModelConfig, PackedCorpus, _, _ = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    seed = TinyLlama(ModelConfig(**checkpoint['config']))
    seed.load_state_dict(checkpoint['model_state_dict'])
    settings = checkpoint['settings']
    corpus = PackedCorpus(args.data_dir, 'train', seed.config.max_seq_len, settings['seed'])
    tokens, targets = corpus.batch(settings['batch_size'], checkpoint['step']+1, 0, 0)
    del checkpoint
    records = []
    for masked in (False, True):
        x, y = tokens.cuda(), targets.cuda()
        mask = None
        if masked:
            mask = torch.ones_like(y, dtype=torch.float32)
            mask[:, :y.shape[1]//2] = 0.
            y[:, :10] = -100
        expected = None
        for mode in ('reference', 'fused', 'vocab-stream'):
            model = copy.deepcopy(seed).cuda()
            assert model.output.weight is model.tok_embeddings.weight
            install(model, shared_projection_casts=True, direct_gqa=True, fused_rope=True, loss_mode=mode)
            assert model.state_dict().keys() == seed.state_dict().keys()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, loss = model(x, y, mask, targets_shifted=not masked,
                                     **({'loss_only': True} if mode != 'reference' else {}))
            loss.backward()
            gradients = {n: p.grad.cpu().clone() for n,p in model.named_parameters()}
            if expected is None:
                expected, expected_loss = gradients, loss.item()
            else:
                assert output is None
                values = {n: difference(g, expected[n]) for n,g in gradients.items()}
                record = dict(mode=mode, masked=masked, loss=loss.item(), reference_loss=expected_loss,
                              loss_abs=abs(loss.item()-expected_loss), gradients=values,
                              tied_embedding=True, parameter_tensors=len(gradients))
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k != 'gradients'}), flush=True)
                assert record['loss_abs'] < 2e-6
                if mode == 'fused':
                    assert all(v['exact'] for v in values.values()), {n:v for n,v in values.items() if not v['exact']}
                else:
                    assert max(v['relative_l2'] for v in values.values()) < .02
                # Default output API, inference and auxiliary-loss fallback.
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    sample = x[:1, :16]
                    logits, _ = model(sample)
                    original, _ = model._nt_original_forward(sample)
                    torch.testing.assert_close(logits, original, rtol=0, atol=0)
                    a = model(sample, sample, loss_only=True, label_smoothing=.1)
                    b = model._nt_original_forward(sample, sample, label_smoothing=.1)
                    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            del model, gradients, loss, output
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root', 'checkpoint', 'data-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.manual_seed(20261014)
    result = dict(primitives=primitives(), semantics=semantics(), model=model_checks(args), status='passed')
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('Target-only loss tests passed', flush=True)


if __name__ == '__main__':
    main()
