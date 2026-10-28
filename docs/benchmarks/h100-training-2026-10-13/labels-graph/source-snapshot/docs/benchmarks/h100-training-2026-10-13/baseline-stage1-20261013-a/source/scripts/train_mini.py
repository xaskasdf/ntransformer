"""Reproducible DDP pilot for packed pretraining and complete-turn SFT.

torchrun --standalone --nproc_per_node=2 scripts/train_mini.py --data-dir ... --output-dir ...
Checkpoints include optimizer, per-rank RNG, data identity and next sampling step.
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mini import PackedCorpus, file_sha256, instruction_batch, read_instruction_jsonl
from src.model import ModelConfig, TinyLlama
from src.tokenizer import Tokenizer


def lr_factor(step, settings):
    warmup, maximum = settings['warmup_steps'], settings['max_steps']
    if step < warmup:
        return (step + 1) / max(1, warmup)
    start = max(warmup, int(maximum * (1 - settings['decay_fraction'])))
    progress = min(1.0, max(0.0, (step - start) / max(1, maximum - start - 1)))
    return settings['min_lr_ratio'] + (1 - settings['min_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizers(model, settings, device):
    matrices, decay, vectors = [], [], []
    for name, parameter in model.named_parameters():
        if settings['optimizer'] == 'muon' and parameter.ndim == 2 and name.startswith('layers.'):
            matrices.append(parameter)
        elif parameter.ndim > 1:
            decay.append(parameter)
        else:
            vectors.append(parameter)
    optimizers = [torch.optim.AdamW([
        {'params': decay, 'weight_decay': settings['weight_decay']},
        {'params': vectors, 'weight_decay': 0.0},
    ], lr=settings['learning_rate'], betas=(0.9, 0.95), fused=device.type == 'cuda')]
    if matrices:
        optimizers.append(torch.optim.Muon(matrices, lr=settings['muon_learning_rate'],
                                           weight_decay=settings['weight_decay']))
    return optimizers


def aggregate(tensor, world_size):
    if world_size > 1:
        dist.all_reduce(tensor)
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model_mini_108m.yaml')
    parser.add_argument('--phase', choices=['pretrain', 'sft'], default='pretrain')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', help='Restore the complete training state')
    parser.add_argument('--init-checkpoint', help='Initialize weights for a new stage')
    parser.add_argument('--stop-after', type=int, help='Stop after this total step without changing the LR schedule')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error('Choose resume or initialization, not both')
    if args.phase == 'sft' and not (args.resume or args.init_checkpoint):
        parser.error('SFT needs a pretrained checkpoint')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=10))
    try:
        run(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args, rank, world, device):
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config['pretrain' if args.phase == 'pretrain' else 'finetune']
    for key in ('max_steps', 'batch_size', 'gradient_accumulation_steps', 'eval_batches',
                'eval_interval', 'save_interval', 'log_interval'):
        if settings[key] < 1:
            raise ValueError(f'{key} must be positive')
    if settings['optimizer'] not in ('adamw', 'muon'):
        raise ValueError('Unsupported optimizer')
    if not 0 <= settings['warmup_steps'] < settings['max_steps']:
        raise ValueError('Warmup must be shorter than the run')
    if not 0 < settings['decay_fraction'] <= 1 or not 0 <= settings['min_lr_ratio'] <= 1:
        raise ValueError('Invalid learning-rate schedule')
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError('stop-after must be positive')
    output = Path(args.output_dir)
    if rank == 0:
        if args.resume:
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()

    root = Path(args.data_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest_sha = file_sha256(root / 'manifest.json')
    tokenizer = Tokenizer(config['tokenizer']['model_path'])
    if file_sha256(config['tokenizer']['model_path']) != manifest['tokenizer_sha256']:
        raise ValueError('Corpus tokenizer does not match model tokenizer')
    if tokenizer.vocab_size != config['model']['vocab_size']:
        raise ValueError('Vocabulary mismatch')
    entries = manifest['sources'] if args.phase == 'pretrain' else [manifest['instructions']]
    for entry in entries:
        for item in entry['files'].values():
            if file_sha256(root / item['path']) != item['sha256']:
                raise ValueError(f'Corpus checksum mismatch: {item["path"]}')
    if args.phase == 'pretrain':
        training = PackedCorpus(root, 'train', config['model']['max_seq_len'], settings['seed'])
        validation = PackedCorpus(root, 'validation', config['model']['max_seq_len'], settings['seed'] + 1)
        groups = validation.names
    else:
        files = manifest['instructions']['files']
        training = read_instruction_jsonl(root / files['train']['path'])
        validation = read_instruction_jsonl(root / files['validation']['path'])
        groups = ['instructions']

    torch.manual_seed(settings['seed'])
    torch.set_float32_matmul_precision('high')
    model = TinyLlama(ModelConfig(**config['model'])).to(device)
    optimizers = make_optimizers(model, settings, device)
    checkpoint = None
    step, initial_validation, best_loss = 0, None, float('inf')
    if args.resume or args.init_checkpoint:
        checkpoint = torch.load(args.resume or args.init_checkpoint, map_location=device, weights_only=False)
        old_config = checkpoint['config']
        if not isinstance(old_config, dict):
            old_config = asdict(old_config)
        if old_config != asdict(model.config):
            raise ValueError('Checkpoint architecture differs')
        if checkpoint.get('tokenizer_sha256') != manifest['tokenizer_sha256']:
            raise ValueError('Checkpoint tokenizer differs or is unknown')
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.resume:
            if checkpoint['settings'] != settings or checkpoint['world_size'] != world or checkpoint['phase'] != args.phase:
                raise ValueError('Exact resume requires the same settings, world size and phase')
            if checkpoint['manifest_sha256'] != manifest_sha:
                raise ValueError('Resume data changed')
            for optimizer, state in zip(optimizers, checkpoint['optimizer_states']):
                optimizer.load_state_dict(state)
            step = checkpoint['step']
            initial_validation = checkpoint['initial_validation']
            best_loss = checkpoint['best_validation_loss']

    distributed = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                      broadcast_buffers=False) if world > 1 else model
    if args.resume:
        state = checkpoint['rng_states'][rank]
        torch.set_rng_state(state['cpu'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    del checkpoint

    def fetch(data, current_step, micro, group=None):
        if args.phase == 'pretrain':
            x, y = data.batch(settings['batch_size'], current_step, rank, micro, source=group)
            return x.to(device), y.to(device), None
        rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], current_step, rank, micro]))
        chosen = rng.integers(len(data), size=settings['batch_size'])
        x, y, mask = instruction_batch([data[i] for i in chosen], tokenizer, model.config.max_seq_len)
        return x.to(device), y.to(device), mask.to(device)

    def forward(candidate, x, y, mask):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            return candidate(x, y, mask, targets_shifted=args.phase == 'pretrain')[1]

    def log(record):
        if rank == 0:
            with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = {}
        for index, name in enumerate(groups):
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for batch_index in range(settings['eval_batches']):
                x, y, mask = fetch(validation, 0, batch_index, index if args.phase == 'pretrain' else None)
                loss = forward(model, x, y, mask)
                count = x.numel() if mask is None else mask[:, 1:].sum().item()
                totals += torch.tensor([loss.item() * count, count], device=device, dtype=torch.float64)
            aggregate(totals, world)
            results[name] = (totals[0] / totals[1]).item()
        model.train()
        # Report the per-source mean explicitly; do not silently weight by corpus size.
        results['mean'] = sum(results.values()) / len(groups)
        if not all(math.isfinite(value) for value in results.values()):
            raise RuntimeError('Nonfinite validation loss')
        log({'event': 'validation', 'step': step, 'loss': results})
        return results

    def save(filename):
        state = {'cpu': torch.get_rng_state(),
                 'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        if rank == 0:
            payload = {'kind': 'brandon-mini-pilot', 'config': asdict(model.config),
                       'model_state_dict': model.state_dict(),
                       'optimizer_states': [optimizer.state_dict() for optimizer in optimizers],
                       'step': step, 'settings': settings, 'world_size': world, 'phase': args.phase,
                       'rng_states': states, 'manifest_sha256': manifest_sha,
                       'tokenizer_sha256': manifest['tokenizer_sha256'],
                       'initial_validation': initial_validation, 'best_validation_loss': best_loss}
            temporary = output / (filename + '.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, output / filename)
        if world > 1:
            dist.barrier()

    if rank == 0:
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        (output / 'data-manifest.json').write_text(json.dumps(manifest, indent=2))
    model.train()
    if initial_validation is None:
        initial_validation = evaluate()
        best_loss = initial_validation['mean']
    maximum = min(settings['max_steps'], args.stop_after or settings['max_steps'])
    if step >= maximum:
        raise ValueError('Checkpoint is already at or beyond the requested stop step')
    start_step, timer, running_loss, running_tokens = step, time.perf_counter(), 0.0, 0
    final_validation = None
    while step < maximum:
        factor = lr_factor(step, settings)
        for index, optimizer in enumerate(optimizers):
            peak = settings['learning_rate'] if index == 0 else settings['muon_learning_rate']
            for group in optimizer.param_groups:
                group['lr'] = peak * factor
            optimizer.zero_grad(set_to_none=True)
        batches = [fetch(training, step, micro) for micro in range(settings['gradient_accumulation_steps'])]
        # Global supervised-token normalization keeps variable-length SFT batches
        # correctly weighted across accumulation steps AND DDP ranks.
        local_count = sum(x.numel() if mask is None else mask[:, 1:].sum().item() for x, _, mask in batches)
        total_count = aggregate(torch.tensor(float(local_count), device=device), world).item()
        if total_count <= 0:
            raise ValueError('No supervised tokens in batch')
        local_sum = 0.0
        for micro, (x, y, mask) in enumerate(batches):
            synchronize = micro == len(batches) - 1
            context = distributed.no_sync() if world > 1 and not synchronize else nullcontext()
            with context:
                loss = forward(distributed, x, y, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at step {step}, rank {rank}')
                count = x.numel() if mask is None else mask[:, 1:].sum().item()
                (loss * (count * world / total_count)).backward()
            local_sum += loss.item() * count
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        running_loss += aggregate(torch.tensor(local_sum, device=device, dtype=torch.float64), world).item()
        running_tokens += total_count
        step += 1
        if step % settings['log_interval'] == 0 or step == maximum:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - timer
            log({'event': 'train', 'step': step, 'loss': running_loss / running_tokens,
                 'supervised_tokens_per_second': running_tokens / seconds,
                 'learning_rate': optimizers[0].param_groups[0]['lr'], 'grad_norm': norm.item()})
            running_loss, running_tokens, timer = 0.0, 0, time.perf_counter()
        if step % settings['eval_interval'] == 0 or step == maximum:
            final_validation = evaluate()
            if final_validation['mean'] < best_loss:
                best_loss = final_validation['mean']
                save('best.pt')
            timer = time.perf_counter()
        if step % settings['save_interval'] == 0 or step == maximum:
            save('last.pt')
            timer = time.perf_counter()
    summary = {'phase': args.phase, 'parameters': model.count_parameters(), 'world_size': world,
               'start_step': start_step, 'completed_steps': step, 'planned_steps': settings['max_steps'],
               'initial_validation': initial_validation, 'final_validation': final_validation,
               'best_validation_loss': best_loss, 'manifest_sha256': manifest_sha,
               'status': 'complete' if step == settings['max_steps'] else 'paused_at_requested_step'}
    if args.phase == 'pretrain':
        summary['training_tokens'] = step * settings['batch_size'] * world * settings['gradient_accumulation_steps'] * model.config.max_seq_len
    if rank == 0:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
