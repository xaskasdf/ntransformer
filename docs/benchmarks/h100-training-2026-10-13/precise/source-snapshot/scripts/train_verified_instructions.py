#!/usr/bin/env python3
"""Paired, bounded SFT using the unchanged Brandon trainer and verified data."""
import argparse
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from train_brandon_experiment import import_brandon, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['reference', 'native', 'norms', 'swiglu'], required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    if not 1 <= config['finetune']['max_steps'] <= 100:
        parser.error('This pilot is bounded to 100 updates')
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        TinyLlama, _, _, file_sha256, reference = import_brandon(args.brandon_root)
        backend = {'mode': args.mode, 'trainer': 'unchanged Brandon train_mini.run',
                   'initialization': 'final SFT weights; new AdamW for synthetic curriculum',
                   'gemm_attention_optimizer_ddp': 'PyTorch'}
        if args.mode != 'reference':
            def factory(model_config):
                model = TinyLlama(model_config)
                backend.update(install(model, native=True, native_norms=args.mode != 'swiglu',
                                       native_swiglu=args.mode != 'norms'))
                return model
            reference.TinyLlama = factory

        batch_records = []
        original_batch = reference.instruction_batch
        def record_batch(*batch_args, **batch_kwargs):
            batch = original_batch(*batch_args, **batch_kwargs)
            digest = hashlib.sha256()
            for tensor in batch:
                digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                digest.update(tensor.contiguous().numpy().tobytes())
            batch_records.append(digest.hexdigest())
            return batch
        reference.instruction_batch = record_batch
        started = time.perf_counter()
        reference.run(SimpleNamespace(config=args.config, phase='sft', data_dir=args.data_dir,
                                      output_dir=args.output, resume=None,
                                      init_checkpoint=args.checkpoint, stop_after=None,
                                      device='cuda'), rank, world, device)
        torch.cuda.synchronize()
        output = Path(args.output)
        (output / f'batches-rank{rank}.json').write_text(json.dumps(batch_records) + '\n')
        if rank == 0:
            backend.update(seconds_including_load_eval_save=time.perf_counter() - started,
                           starting_checkpoint_sha256=file_sha256(args.checkpoint),
                           trainer_sha256=file_sha256(Path(args.brandon_root) / 'scripts/train_mini.py'),
                           model_source_sha256=file_sha256(Path(args.brandon_root) / 'src/model.py'),
                           torch=torch.__version__, cuda_build=torch.version.cuda,
                           batch_record_scope='Every train and validation fetch, in order, per rank')
            (output / 'experiment.json').write_text(json.dumps(backend, indent=2) + '\n')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
