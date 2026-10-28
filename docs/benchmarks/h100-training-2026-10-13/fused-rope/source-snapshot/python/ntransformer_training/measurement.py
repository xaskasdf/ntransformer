"""CPU-only comparison contract and timing summaries for training experiments."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_baseline(checkpoint, baseline_path, source_root, parameters, software):
    baseline_path = Path(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    hashes = json.loads((baseline_path.parent / 'SHA256SUMS.json').read_text())
    digest = file_sha256(baseline_path)
    if hashes['baseline.json'] != digest:
        raise ValueError('Historical baseline checksum mismatch')
    if parameters != baseline['parameters'] or checkpoint['world_size'] != baseline['world_size']:
        raise ValueError('Model size or original world size differs from the baseline')
    for key, expected in baseline['config']['model'].items():
        actual = checkpoint['config'][key]
        if key in ('norm_eps', 'rope_theta', 'dropout'):
            actual, expected = float(actual), float(expected)
        if actual != expected:
            raise ValueError(f'Architecture differs from baseline: {key}')
    if checkpoint['settings'] != baseline['config']['pretrain']:
        raise ValueError('Training settings differ from the historical baseline')
    captured = baseline['captured_file_hashes']
    if checkpoint['manifest_sha256'] != captured['data-manifest.json']['sha256']:
        raise ValueError('Corpus identity differs from the historical baseline')
    for relative in ('src/model.py', 'src/data/mini.py', 'src/tokenizer.py', 'scripts/train_mini.py'):
        if file_sha256(Path(source_root) / relative) != captured['source/' + relative]['sha256']:
            raise ValueError(f'Brandon source differs from the historical baseline: {relative}')
    environment = baseline['environment']
    if (software['torch'].split('+')[0] != environment['packages']['torch']
            or software['cuda_build'] != environment['torch_cuda']
            or list(software['nccl']) != environment['nccl']):
        raise ValueError('PyTorch/CUDA build/NCCL differs from the historical baseline')
    return dict(baseline_sha256=digest, captured_step=baseline['latest_train']['step'],
                historical_tokens_per_second=baseline['timing']['last_1000_steps']['aggregate_tokens_per_second'],
                scope='Historical context; paired continuation is the causal implementation comparison',
                initialization='Completed pretraining checkpoint, restored AdamW, final LR floor; not a pilot restart')


def window_summary(records, window_seconds):
    measured = [record for record in records if record['measured']]
    if not measured or window_seconds <= 0:
        raise ValueError('Need a positive measured window with at least one step')
    tokens = sum(record['tokens'] for record in measured)
    # Aggregate tokens divided by elapsed window, never arithmetic mean of rates.
    return dict(tokens_per_second=tokens / window_seconds,
                mean_step_seconds=window_seconds / len(measured),
                measured_window_seconds=window_seconds,
                instrumented_step_mean_seconds=sum(record['seconds'] for record in measured) / len(measured))
