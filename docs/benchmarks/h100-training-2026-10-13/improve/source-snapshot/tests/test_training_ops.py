#!/usr/bin/env python3
"""GPU numerical tests, including independent FP64 derivatives and full-model gradients."""
import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ntransformer_training.ops import rmsnorm, swiglu
from ntransformer_training.brandon import install
from ntransformer_training.pipeline import PeerActivationPool
from train_brandon_experiment import import_brandon


def relative(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12)).item()


def primitives():
    torch.manual_seed(1701)
    records = []
    # Odd/tiny/wide dimensions and a multi-part weight reduction.
    for rows, dim in [(1, 1), (3, 33), (257, 576), (9, 4097)]:
        x = torch.randn(rows, dim, device='cuda', requires_grad=True)
        w = torch.randn(dim, device='cuda', requires_grad=True)
        dy = torch.randn_like(x)
        y = rmsnorm(x, w, 1e-5)
        gx, gw = torch.autograd.grad(y, (x, w), dy)
        xd, wd = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
        expected = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + 1e-5) * wd
        ex, ew = torch.autograd.grad(expected, (xd, wd), dy.double())
        torch.testing.assert_close(y.double(), expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(gx.double(), ex, rtol=3e-4, atol=4e-6)
        torch.testing.assert_close(gw.double(), ew, rtol=3e-4, atol=2e-5)
        records.append(dict(op='rmsnorm', shape=[rows, dim], dx_relative=relative(gx, ex), dw_relative=relative(gw, ew)))
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.randn(7, 1472, device='cuda', dtype=dtype, requires_grad=True)
        u = torch.randn_like(g, requires_grad=True)
        dy = torch.randn_like(g)
        out = swiglu(g, u)
        expected = F.silu(g) * u
        native = torch.autograd.grad(out, (g, u), dy)
        reference = torch.autograd.grad(expected, (g, u), dy)
        tolerance = 8e-3 if dtype == torch.bfloat16 else 3e-6
        torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
        for result, target in zip(native, reference):
            torch.testing.assert_close(result, target, rtol=tolerance, atol=tolerance)
            if relative(result, target) > tolerance:
                raise AssertionError('SwiGLU aggregate gradient error exceeds tolerance')
        records.append(dict(op='swiglu', dtype=str(dtype), dx_relative=[relative(a, b) for a, b in zip(native, reference)]))
    # Captured from real SFT: unit-scale random gradients miss these BF16 ties.
    # Inference's --use_fast_math incorrectly changed expf in this backward.
    g = torch.full((3,), -2.09375, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    u = torch.tensor([-.82421875, .4375, -.93359375], device='cuda',
                     dtype=torch.bfloat16, requires_grad=True)
    dy = torch.tensor([-4.3392181396484375e-5, -1.0251998901367188e-5,
                       -4.685716703534126e-9], device='cuda', dtype=torch.bfloat16)
    actual, expected = swiglu(g, u), F.silu(g) * u
    native = torch.autograd.grad(actual, (g, u), dy)
    reference = torch.autograd.grad(expected, (g, u), dy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(native, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    records.append(dict(op='swiglu_small_gradient_rounding', exact=True, cases=3))
    # Finite differences check a subset of input and norm weight coordinates.
    x = torch.randn(2, 8, device='cuda', requires_grad=True)
    w = torch.randn(8, device='cuda', requires_grad=True)
    probe = torch.randn_like(x)
    grad = torch.autograd.grad((rmsnorm(x, w, 1e-3) * probe).sum(), (x, w))
    for tensor, analytic in zip((x, w), grad):
        for index in (0, tensor.numel() - 1):
            original = tensor.detach().flatten()[index].item()
            values = []
            for delta in (1e-3, -1e-3):
                with torch.no_grad():
                    tensor.flatten()[index] = original + delta
                    values.append((rmsnorm(x, w, 1e-3) * probe).sum().item())
            with torch.no_grad():
                tensor.flatten()[index] = original
            numeric = (values[0] - values[1]) / 2e-3
            if abs(numeric - analytic.flatten()[index].item()) > 3e-3:
                raise AssertionError('RMSNorm finite difference mismatch')
    # Current stream and non-contiguous output gradients must both work.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(3, 576, device='cuda', requires_grad=True)
        w = torch.ones(576, device='cuda', requires_grad=True)
        rmsnorm(x, w, 1e-5).sum().backward()
    stream.synchronize()
    if not torch.isfinite(x.grad).all():
        raise AssertionError('Stream test failed')
    for bad in (torch.ones(2, 4), torch.ones(2, 4, device='cuda', dtype=torch.bfloat16)):
        try:
            rmsnorm(bad, torch.ones(4, device=bad.device), 1e-5)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported RMSNorm input was accepted')
    return records


def model_tests(args):
    TinyLlama, ModelConfig, _, _, trainer = import_brandon(args.brandon_root)
    if args.cpu:
        config = ModelConfig(dim=32, n_layers=3, n_heads=4, n_kv_heads=2,
                             vocab_size=64, hidden_dim=64, max_seq_len=16, dropout=0)
        seed_model = TinyLlama(config)
        batch, length, device = 2, 8, torch.device('cpu')
        settings = dict(optimizer='adamw', learning_rate=3e-5, weight_decay=.1)
    else:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = ModelConfig(**ckpt['config'])
        seed_model = TinyLlama(config)
        seed_model.load_state_dict(ckpt['model_state_dict'])
        settings = ckpt['settings']
        del ckpt
        batch, length, device = 1, 64, torch.device('cuda', 0)
    torch.manual_seed(444)
    x = torch.randint(config.vocab_size, (batch, length), device=device)
    y = torch.randint(config.vocab_size, (batch, length), device=device)
    results = []
    precisions = [False] if args.cpu else [False, True]
    for bf16 in precisions:
        reference = copy.deepcopy(seed_model).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            reference_loss = reference(x, y, targets_shifted=True)[1]
        reference_loss.backward()
        reference_grads = {name: p.grad.detach().cpu().clone() for name, p in reference.named_parameters()}
        reference_optimizer = trainer.make_optimizers(reference, settings, device)[0]
        reference_optimizer.step()
        reference_weights = {name: p.detach().cpu().clone() for name, p in reference.named_parameters()}
        expected_loss = reference_loss.item()
        del reference, reference_optimizer, reference_loss
        checkpoint_grads = None
        modes = ['checkpoint'] if args.cpu else ['native', 'checkpoint', 'peer']
        for mode in modes:
            candidate = copy.deepcopy(seed_model).to(device)
            pool = None
            if mode == 'peer':
                pool = PeerActivationPool(0, 1, config.n_layers, (batch, length, config.dim))
            install(candidate, native=not args.cpu, checkpoint=mode != 'native', peer_pool=pool,
                    norm_mode=args.norm_mode)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                loss = candidate(x, y, targets_shifted=True)[1]
            loss.backward()
            if pool:
                pool.finish()
            squared_error = squared_reference = 0.0
            worst = 0.0
            for name, parameter in candidate.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise AssertionError(f'Missing/nonfinite gradient: {name}')
                actual = parameter.grad.detach().cpu()
                target = reference_grads[name]
                squared_error += (actual.double() - target.double()).square().sum().item()
                squared_reference += target.double().square().sum().item()
                worst = max(worst, relative(actual, target))
                if mode == 'peer':
                    torch.testing.assert_close(actual, checkpoint_grads[name], rtol=1e-5, atol=1e-7)
            if mode == 'checkpoint' and not args.cpu:
                checkpoint_grads = {name: p.grad.detach().cpu().clone() for name, p in candidate.named_parameters()}
            error = (squared_error / squared_reference) ** .5
            tolerance = 0.03 if bf16 else (2e-3 if not args.cpu else 1e-6)
            if error > tolerance or worst > tolerance * 3 or abs(loss.item() - expected_loss) > (0.01 if bf16 else 1e-4):
                raise AssertionError(f'{mode} gradient/loss mismatch: L2={error}, worst={worst}, loss={loss.item()-expected_loss}')
            optimizer = trainer.make_optimizers(candidate, settings, device)[0]
            optimizer.step()
            update_error = max((p.detach().cpu() - reference_weights[name]).abs().max().item()
                               for name, p in candidate.named_parameters())
            # Small near-zero gradients may flip Adam's first-step sign.
            if update_error > 2.1 * settings['learning_rate']:
                raise AssertionError('AdamW update exceeds the allowed first-step rounding bound')
            if candidate.output.weight is not candidate.tok_embeddings.weight:
                raise AssertionError('Tied weights were lost')
            results.append(dict(mode=mode, bf16=bf16, loss=loss.item(), reference_loss=expected_loss,
                                gradient_relative_l2=error, worst_parameter_relative_l2=worst,
                                update_max_abs=update_error, gradient_tolerance=tolerance,
                                pipeline=pool.statistics() if pool else None))
            # A second backward tests event/generation reuse and accumulation.
            if pool:
                candidate.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    candidate(x, y, targets_shifted=True)[1].backward()
                pool.finish()
                pool.synchronize()
                if pool.copies_out != 2 * config.n_layers or pool.copies_in != pool.copies_out:
                    raise AssertionError('Peer transfer accounting mismatch')
            del candidate, optimizer, pool, loss
            gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--norm-mode', choices=['fused', 'compatible'], default='fused')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    if not args.cpu and torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs are required; this validation may not silently skip P2P')
    result = dict(status='passed', primitive_tests=[] if args.cpu else primitives(), model_tests=model_tests(args))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
