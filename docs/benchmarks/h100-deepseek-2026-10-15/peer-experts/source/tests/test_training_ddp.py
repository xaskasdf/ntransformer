#!/usr/bin/env python3
"""CPU/Gloo test for nested block recomputation, DDP and gradient accumulation."""
import argparse
import copy
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ntransformer_training.brandon import install
from train_brandon_experiment import import_brandon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        TinyLlama, ModelConfig, _, _, _ = import_brandon(args.brandon_root)
        torch.manual_seed(2026)
        seed_model = TinyLlama(ModelConfig(dim=32, n_layers=3, n_heads=4, n_kv_heads=2,
                               vocab_size=64, hidden_dim=64, max_seq_len=16, dropout=0))
        gradients = []
        for checkpoint in (False, True):
            model = copy.deepcopy(seed_model)
            if checkpoint:
                install(model, native=False, checkpoint=True)
            ddp = DDP(model, broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                for micro in range(2):
                    generator = torch.Generator().manual_seed(1000 + rank * 100 + step * 2 + micro)
                    x = torch.randint(64, (2, 8), generator=generator)
                    y = torch.randint(64, (2, 8), generator=generator)
                    with ddp.no_sync() if micro == 0 else nullcontext():
                        (ddp(x, y, targets_shifted=True)[1] / 2).backward()
                optimizer.step()
            gradients.append({name: (p.detach().clone(), p.grad.clone()) for name, p in model.named_parameters()})
        max_weight_error = max_grad_error = 0.0
        for name, (weight, grad) in gradients[0].items():
            target_weight, target_grad = gradients[1][name]
            torch.testing.assert_close(weight, target_weight, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(grad, target_grad, rtol=1e-5, atol=1e-7)
            max_weight_error = max(max_weight_error, (weight - target_weight).abs().max().item())
            max_grad_error = max(max_grad_error, (grad - target_grad).abs().max().item())
        record = dict(status='passed', rank=rank, world_size=world, steps=3,
                      weight_max_abs=max_weight_error, gradient_max_abs=max_grad_error)
        records = [None] * world
        dist.all_gather_object(records, record)
        if rank == 0:
            Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(records), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
