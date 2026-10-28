#!/usr/bin/env python3
"""Bounded continuation from a completed Brandon pretraining checkpoint.

Run through queue_brandon_training.py to enforce the existing pipeline's gate.
All modes restore the same optimizer state and deterministic corpus sampling.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from ntransformer_training.brandon import install, validate_config
from ntransformer_training.pipeline import PeerActivationPool
from ntransformer_training.measurement import validate_baseline, window_summary


def import_brandon(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from src.model import TinyLlama, ModelConfig
    from src.data.mini import PackedCorpus, file_sha256
    spec = importlib.util.spec_from_file_location('brandon_reference_trainer', Path(root) / 'scripts/train_mini.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return TinyLlama, ModelConfig, PackedCorpus, file_sha256, trainer


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['reference', 'native', 'checkpoint', 'peer'], required=True)
    parser.add_argument('--norm-mode', choices=['fused', 'compatible'], default='compatible')
    parser.add_argument('--deterministic', action='store_true',
                        help='Diagnostic run: enforce deterministic algorithms, including SDPA backward')
    parser.add_argument('--skip-uninitialized-fill', action='store_true',
                        help='Audited deterministic experiment: omit debug fills before fully written outputs')
    parser.add_argument('--gradient-bucket-views', action='store_true',
                        help='Keep DDP gradients as bucket views to avoid copying reduced gradients back')
    parser.add_argument('--defer-loss-readbacks', action='store_true',
                        help='Check losses together after accumulation, before clipping or updating weights')
    parser.add_argument('--shared-projection-casts', action='store_true',
                        help='Share Q/K/V and gate/up BF16 inputs with separate FP32 backward edges')
    parser.add_argument('--direct-gqa', action='store_true',
                        help='Pass compact KV heads to SDPA instead of explicit head replication')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--peer-device', type=int)
    parser.add_argument('--historical-baseline', default=str(Path(__file__).resolve().parents[1] /
                        'docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a/baseline.json'))
    parser.add_argument('--profile', action='store_true', help='Separate diagnostic mode; excludes profiler from throughput claims')
    parser.add_argument('--profile-start', type=int, default=0,
                        help='Skip this many update offsets before profiling two warmed updates')
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 0 <= args.warmup_steps < args.steps:
        parser.error('This experiment is bounded to 1..100 steps; warmup must be shorter')
    if args.skip_uninitialized_fill and not args.deterministic:
        parser.error('--skip-uninitialized-fill requires --deterministic')
    if (args.shared_projection_casts or args.direct_gqa) and args.mode == 'reference':
        parser.error('Projection/attention optimizations require a native adapter mode')
    if args.profile_start < 0 or (args.profile_start and
            (not args.profile or args.steps != args.profile_start + 2)):
        parser.error('--profile-start requires --profile and exactly two trailing active steps')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=False)
        if args.skip_uninitialized_fill:
            torch.utils.deterministic.fill_uninitialized_memory = False
    rank, local, world = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1)])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        run(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, rank, world, device):
    TinyLlama, ModelConfig, PackedCorpus, file_sha256, reference = import_brandon(args.brandon_root)
    output = Path(args.output).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    config = config if isinstance(config, dict) else asdict(config)
    settings = checkpoint['settings']
    logical_world = checkpoint['world_size']
    if checkpoint['phase'] != 'pretrain' or checkpoint['step'] != settings['max_steps']:
        raise ValueError('A completed pretraining last.pt is required')
    if settings['optimizer'] != 'adamw' or world not in (1, logical_world):
        raise ValueError('Only AdamW with the original DDP size or equivalent single-GPU batches is supported')
    manifest = json.loads((Path(args.data_dir) / 'manifest.json').read_text())
    if (file_sha256(Path(args.data_dir) / 'manifest.json') != checkpoint['manifest_sha256']
            or manifest['tokenizer_sha256'] != checkpoint['tokenizer_sha256']):
        raise ValueError('Checkpoint and corpus identities differ')
    torch.manual_seed(settings['seed'])
    torch.set_float32_matmul_precision('high')
    model = TinyLlama(ModelConfig(**config)).to(device)
    validate_config(model.config)
    historical = validate_baseline(dict(checkpoint, config=config), args.historical_baseline,
                                   args.brandon_root, model.count_parameters(),
                                   dict(torch=torch.__version__, cuda_build=torch.version.cuda,
                                        nccl=torch.cuda.nccl.version()))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizers = reference.make_optimizers(model, settings, device)
    if len(optimizers) != len(checkpoint['optimizer_states']):
        raise ValueError('Optimizer state count differs')
    for optimizer, state in zip(optimizers, checkpoint['optimizer_states']):
        optimizer.load_state_dict(state)
        for group in optimizer.param_groups:
            group['lr'] = settings['learning_rate'] * reference.lr_factor(checkpoint['step'], settings)
    pool = None
    if args.mode == 'peer':
        peer = args.peer_device if args.peer_device is not None else 1 - device.index
        pool = PeerActivationPool(device.index, peer, model.config.n_layers,
                                  (settings['batch_size'], model.config.max_seq_len, model.config.dim))
    backend = dict(gemm_attention_optimizer_ddp='PyTorch', native_norms=0, native_swiglu=0,
                   checkpoint_layers=0, peer_activation_storage=False)
    if args.mode != 'reference':
        backend = install(model, native=True, checkpoint=args.mode in ('checkpoint', 'peer'),
                          peer_pool=pool, norm_mode=args.norm_mode,
                          shared_projection_casts=args.shared_projection_casts, direct_gqa=args.direct_gqa)
    backend.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                   gradient_bucket_views=args.gradient_bucket_views,
                   defer_loss_readbacks=args.defer_loss_readbacks,
                   cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                   attention_dispatch='PyTorch SDPA automatic; constrained by determinism setting')
    distributed = (DDP(model, device_ids=[device.index], broadcast_buffers=False,
                       gradient_as_bucket_view=args.gradient_bucket_views) if world > 1 else model)
    state = checkpoint['rng_states'][rank]
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], device)
    start_step = checkpoint['step']
    initial_best = checkpoint['best_validation_loss']
    del checkpoint
    train = PackedCorpus(args.data_dir, 'train', model.config.max_seq_len, settings['seed'])
    validation = PackedCorpus(args.data_dir, 'validation', model.config.max_seq_len, settings['seed'] + 1)
    logical_ranks = [rank] if world > 1 else range(logical_world)
    schedule = [(logical_rank, micro) for micro in range(settings['gradient_accumulation_steps'])
                for logical_rank in logical_ranks]

    def reduce(tensor):
        if world > 1:
            dist.all_reduce(tensor)
        return tensor

    @torch.no_grad()
    def evaluate():
        model.eval()
        values = {}
        for group, name in enumerate(validation.names):
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for index in range(settings['eval_batches']):
                for logical_rank in logical_ranks:
                    x, y = validation.batch(settings['batch_size'], 0, logical_rank, index, source=group)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(x.to(device), y.to(device), targets_shifted=True)[1]
                    totals += torch.tensor([loss.item() * x.numel(), x.numel()], device=device, dtype=torch.float64)
            reduce(totals)
            values[name] = (totals[0] / totals[1]).item()
        values['mean'] = sum(values.values()) / len(values)
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise RuntimeError('Nonfinite validation')
        model.train()
        return values

    timings = dict(initialization_seconds=time.perf_counter() - PROCESS_STARTED)
    phase_start = time.perf_counter()
    initial_validation = evaluate()
    timings['initial_validation_seconds'] = time.perf_counter() - phase_start
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    records, batch_hashes = [], []
    training_started = time.perf_counter()
    measurement_started = None
    profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA],
                                       record_shapes=True, profile_memory=True,
                                       schedule=(torch.profiler.schedule(wait=args.profile_start - 1,
                                                                        warmup=1, active=2, repeat=1)
                                                 if args.profile_start else None))
                if args.profile else nullcontext())
    with profiler as prof:
        for offset in range(args.steps):
            step = start_step + offset
            torch.cuda.synchronize(device)
            if offset == args.warmup_steps:
                if world > 1:
                    dist.barrier()
                measurement_started = time.perf_counter()
            started = time.perf_counter()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            local_loss = 0.0
            local_tokens = 0
            pending_losses = []
            phase_events = []
            for index, (logical_rank, micro) in enumerate(schedule):
                with torch.profiler.record_function('nt::sample_batch') if args.profile else nullcontext():
                    x, y = train.batch(settings['batch_size'], step, logical_rank, micro)
                with torch.profiler.record_function('nt::hash_batch') if args.profile else nullcontext():
                    digest = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
                batch_hashes.append([step, logical_rank, micro, digest])
                context = distributed.no_sync() if world > 1 and index + 1 < len(schedule) else nullcontext()
                with context:
                    forward_start, forward_end, backward_start, backward_end = [
                        torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    forward_start.record()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = distributed(x.to(device), y.to(device), targets_shifted=True)[1]
                    forward_end.record()
                    if not args.defer_loss_readbacks and not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss at {step}')
                    backward_start.record()
                    (loss / len(schedule)).backward()
                    backward_end.record()
                    phase_events.append((forward_start, forward_end, backward_start, backward_end))
                if pool:
                    pool.finish()
                if args.defer_loss_readbacks:
                    pending_losses.append((loss.detach(), x.numel()))
                else:
                    local_loss += loss.item() * x.numel()
                local_tokens += x.numel()
            if args.defer_loss_readbacks:
                # Preserve Python FP64 accumulation and microbatch order exactly.
                # One vector readback replaces individual mid-microbatch fences.
                values = torch.stack([loss for loss, _ in pending_losses]).tolist()
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f'Nonfinite loss at {step}')
                local_loss = sum(value * count for value, (_, count) in zip(values, pending_losses))
            optimizer_start, optimizer_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            optimizer_start.record()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            for optimizer in optimizers:
                optimizer.step()
            optimizer_end.record()
            totals = reduce(torch.tensor([local_loss, local_tokens], device=device, dtype=torch.float64))
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            if world > 1:
                maximum = torch.tensor(seconds, device=device, dtype=torch.float64)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                seconds = maximum.item()
            record = dict(step=step + 1, loss=(totals[0] / totals[1]).item(), seconds=seconds,
                          tokens=int(totals[1].item()), grad_norm=norm.item(),
                          learning_rate=optimizers[0].param_groups[0]['lr'],
                          forward_stream_ms=sum(a.elapsed_time(b) for a, b, _, _ in phase_events),
                          backward_stream_ms=sum(a.elapsed_time(b) for _, _, a, b in phase_events),
                          optimizer_stream_ms=optimizer_start.elapsed_time(optimizer_end),
                          measured=offset >= args.warmup_steps)
            records.append(record)
            with (output / f'metrics-rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if rank == 0:
                with (output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                if (offset + 1) % 10 == 0 or offset + 1 == args.steps:
                    print(json.dumps(record), flush=True)
            if args.profile:
                prof.step()
    torch.cuda.synchronize(device)
    measured_window_seconds = time.perf_counter() - measurement_started
    timings['training_seconds_including_warmup'] = time.perf_counter() - training_started
    if world > 1:
        maximum = torch.tensor(measured_window_seconds, device=device, dtype=torch.float64)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        measured_window_seconds = maximum.item()
    if args.profile:
        prof.export_chrome_trace(str(output / f'trace-rank{rank}.json'))
        (output / f'profile-rank{rank}.txt').write_text(prof.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=60))
    measured = [record for record in records if record['measured']]
    memory = [dict(device=index, peak_allocated_bytes=torch.cuda.max_memory_allocated(index),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(index))
              for index in range(torch.cuda.device_count())]
    rank_result = dict(rank=rank, batches=batch_hashes, memory=memory,
                       pipeline=pool.statistics() if pool else None)
    rank_results = [None] * world
    if world > 1:
        dist.all_gather_object(rank_results, rank_result)
    else:
        rank_results[0] = rank_result
    phase_start = time.perf_counter()
    final_validation = evaluate()
    timings['final_validation_seconds'] = time.perf_counter() - phase_start
    if pool:
        pool.synchronize()
    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
    rngs = [None] * world
    if world > 1:
        dist.all_gather_object(rngs, rng)
    else:
        rngs[0] = rng
    if rank == 0:
        phase_start = time.perf_counter()
        batches = sorted(batch for result in rank_results for batch in result.pop('batches'))
        write_json(output / 'batch_hashes.json', batches)
        batch_digest = hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest()
        payload = dict(kind='ntransformer-training-experiment-v1', config=asdict(model.config),
                       model_state_dict=model.state_dict(), optimizer_states=[o.state_dict() for o in optimizers],
                       step=start_step + args.steps, settings=settings, world_size=world,
                       logical_world_size=logical_world, phase='pretrain', rng_states=rngs,
                       manifest_sha256=file_sha256(Path(args.data_dir) / 'manifest.json'),
                       tokenizer_sha256=manifest['tokenizer_sha256'],
                       initial_validation=initial_validation, best_validation_loss=min(initial_best, final_validation['mean']),
                       experiment_backend=backend, source_checkpoint=str(Path(args.checkpoint).resolve()))
        torch.save(payload, output / 'last.pt.tmp')
        (output / 'last.pt.tmp').replace(output / 'last.pt')
        timings['checkpoint_and_batch_record_seconds'] = time.perf_counter() - phase_start
        timings['process_seconds_through_checkpoint'] = time.perf_counter() - PROCESS_STARTED
        summary = dict(status='complete', mode=args.mode, backend=backend, steps=args.steps,
                       start_step=start_step, completed_steps=start_step + args.steps,
                       physical_world_size=world, logical_world_size=logical_world,
                       global_tokens_per_step=records[0]['tokens'], parameters=model.count_parameters(),
                       warmup_steps=args.warmup_steps, measured_steps=len(measured),
                       **window_summary(records, measured_window_seconds),
                       timings_rank0=timings, historical_baseline=historical,
                       initial_validation=initial_validation, final_validation=final_validation,
                       batch_sha256=batch_digest, ranks=rank_results, profiled=args.profile,
                       torch_version=torch.__version__, cuda_version=torch.version.cuda,
                       nccl_version=list(torch.cuda.nccl.version()),
                       source_checkpoint=str(Path(args.checkpoint).resolve()))
        write_json(output / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.barrier()


if __name__ == '__main__':
    main()
