#!/usr/bin/env python3
"""Exact branch-rounding, lifetime and full Brandon gradient checks on CUDA."""
import argparse
import copy
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ntransformer_training.projections import _BranchCast, share_projection_casts
from train_brandon_experiment import import_brandon, install


def equal(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def branch_rounding():
    records = []
    for branches in (2, 3):
        x = torch.randn(2, 7, 576, device='cuda', requires_grad=True)
        gradients = [torch.full_like(x, value, dtype=torch.bfloat16)
                     for value in (1., .001, .001)[:branches]]
        reference = [x.to(torch.bfloat16) for _ in range(branches)]
        converted = x.detach().to(torch.bfloat16)
        shared = [_BranchCast.apply(x, converted) for _ in range(branches)]
        assert len({v.data_ptr() for v in shared}) == 1
        assert len({v.grad_fn for v in shared}) == branches
        expected, = torch.autograd.grad(reference, x, gradients)
        actual, = torch.autograd.grad(shared, x, gradients)
        equal(actual, expected)
        naive = x.to(torch.bfloat16)
        naive_gradient, = torch.autograd.grad([naive] * branches, x, gradients)
        assert not torch.equal(naive_gradient, expected), 'Test must detect BF16 accumulation'
        records.append(dict(branches=branches, exact=True, shared_storage=True,
                            naive_max_error=(naive_gradient - expected).abs().max().item()))
    return records


class ProjectionGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(32, 48, bias=False)
        self.k = torch.nn.Linear(32, 16, bias=False)
        self.v = torch.nn.Linear(32, 16, bias=False)

    def forward(self, x, fail=False, leaf_input=False):
        # Brandon projections consume a norm output, not a trainable leaf.
        x = x if leaf_input else x * 1.
        q = self.q(x)
        if fail:
            raise RuntimeError('intentional scope cleanup test')
        return q, self.k(x), self.v(x)


def group_checks():
    original = ProjectionGroup().cuda()
    shared = copy.deepcopy(original)
    share_projection_casts(shared, ('q', 'k', 'v'))
    assert original.state_dict().keys() == shared.state_dict().keys()
    records = []
    for name, autocast, grad_context in (
            ('bf16', True, nullcontext), ('bf16_leaf', True, nullcontext),
            ('fp32', False, nullcontext),
            ('no_grad', True, torch.no_grad), ('inference', True, torch.inference_mode)):
        x = torch.randn(2, 7, 32, device='cuda', requires_grad=True)
        with grad_context(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            left = original(x, leaf_input=name == 'bf16_leaf')
            right = shared(x, leaf_input=name == 'bf16_leaf')
            # A second forward must not invalidate the first one's saved storage.
            left2, right2 = original(x * .25), shared(x * .25)
        assert shared._nt_cast_scope.current is None
        for a, b in zip(left + left2, right + right2):
            equal(a, b)
        if name in ('bf16', 'bf16_leaf', 'fp32'):
            probes = [torch.randn_like(v) * 1e-4 for v in left + left2]
            ga = torch.autograd.grad(left + left2, (x, *original.parameters()), probes)
            gb = torch.autograd.grad(right + right2, (x, *shared.parameters()), probes)
            for a, b in zip(ga, gb):
                equal(a, b)
        records.append(dict(case=name, exact=True))
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            shared(torch.randn(2, 7, 32, device='cuda'), fail=True)
    except RuntimeError as error:
        assert str(error) == 'intentional scope cleanup test'
    else:
        raise AssertionError('Expected intentional exception')
    assert shared._nt_cast_scope.current is None
    # Deepcopy must preserve scope sharing within the clone, not across models.
    clone = copy.deepcopy(shared)
    assert clone._nt_cast_scope is clone.q._nt_cast_scope
    assert clone._nt_cast_scope is not shared._nt_cast_scope
    cpu = copy.deepcopy(original).cpu()
    cpu_shared = copy.deepcopy(cpu)
    share_projection_casts(cpu_shared, ('q', 'k', 'v'))
    sample = torch.randn(2, 7, 32)
    for a, b in zip(cpu(sample), cpu_shared(sample)):
        equal(a, b)
    return records


def model_checks(args, *, direct_gqa=False, fused_rope=False):
    TinyLlama, ModelConfig, PackedCorpus, _, _ = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    seed = TinyLlama(ModelConfig(**checkpoint['config']))
    seed.load_state_dict(checkpoint['model_state_dict'])
    settings = checkpoint['settings']
    corpus = PackedCorpus(args.data_dir, 'train', seed.config.max_seq_len, settings['seed'])
    tokens, targets = corpus.batch(settings['batch_size'], checkpoint['step'] + 1, 0, 0)
    del checkpoint
    records = []
    for masked in (False, True):
        x, y = tokens.cuda(), targets.cuda()
        if masked:
            y[:, :y.shape[1] // 2] = -100
        expected = None
        for enabled in (False, True):
            model = copy.deepcopy(seed).cuda()
            install(model, shared_projection_casts=enabled, direct_gqa=enabled and direct_gqa,
                    fused_rope=enabled and fused_rope)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, y, targets_shifted=True)[1]
            loss.backward()
            gradients = {name: p.grad.cpu().clone() for name, p in model.named_parameters()}
            if expected is None:
                expected, expected_loss = gradients, loss.item()
            else:
                assert loss.item() == expected_loss
                for name, value in gradients.items():
                    equal(value, expected[name])
                assert all(layer.attention._nt_cast_scope.current is None for layer in model.layers)
            del model, gradients, loss
        records.append(dict(masked_targets=masked, loss=expected_loss,
                            exact_parameter_gradients=len(expected), parameters=seed.count_parameters()))
        print(json.dumps(records[-1]), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root', 'checkpoint', 'data-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.manual_seed(20261014)
    result = dict(branch_rounding=branch_rounding(), groups=group_checks())
    result['model'] = model_checks(args)
    result['status'] = 'passed'
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
