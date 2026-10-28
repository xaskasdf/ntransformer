#!/usr/bin/env python3
"""Train 100 paired SFT steps and score exact instruction following on H100.

Run after the performance experiment, with its output passed explicitly. No
teacher outputs are regenerated or repaired. All validation requests are scored.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--distillation', required=True)
    parser.add_argument('--performance', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--native-mode', choices=['swiglu', 'native'], default='native',
                        help='Native uses SwiGLU and compatible RMSNorm; swiglu keeps reference RMSNorm')
    parser.add_argument('--norm-mode', choices=['compatible', 'fused'], default='compatible',
                        help='Compatible preserves PyTorch reductions; fused is a diagnostic alternative')
    args = parser.parse_args()
    root, pipeline, data, performance, output = map(Path, (
        args.brandon_root, args.pipeline, args.distillation, args.performance, args.output))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    status = {'state': 'starting', 'pid': os.getpid(), 'steps_per_backend': 100}
    def update(**values):
        status.update(values, updated_unix=time.time())
        write(output / 'status.json', status)
    def execute(name, command, env, timeout=1800):
        update(state='running', stage=name, command=list(map(str, command)))
        with (output / 'logs' / (name + '.log')).open('w') as handle:
            # start_new_session limits timeout termination to this child group.
            process = subprocess.Popen(list(map(str, command)), env=env, cwd=root,
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                result = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if result:
                raise RuntimeError(f'{name} failed with status {result}; see its log')
    lock = None
    try:
        update(state='waiting', stage='performance-completion')
        deadline = time.monotonic() + 3600
        while read(performance / 'status.json')['state'] == 'running':
            if time.monotonic() > deadline:
                raise TimeoutError('Performance experiment did not finish')
            time.sleep(10)
        if read(performance / 'status.json')['state'] != 'complete':
            raise RuntimeError('Require successful numerical/performance experiment first')
        if read(pipeline / 'status.json')['state'] != 'complete':
            raise RuntimeError('Original pipeline must be complete')
        lock = (pipeline / 'pipeline.lock').open('a')
        # The performance supervisor marks completion before flushing its monitor.
        # Allow its finally block to release the shared pipeline lock.
        lock_deadline = time.monotonic() + 60
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > lock_deadline:
                    raise
                time.sleep(2)
        for _ in range(2):
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                              '--format=csv,noheader,nounits'], text=True).strip()
            if active:
                raise RuntimeError(f'GPU processes already active: {active}')
            time.sleep(2)
        import torch
        import yaml
        checkpoint = pipeline / 'sft/best.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        from dataclasses import asdict
        model_config = payload['config']
        if not isinstance(model_config, dict):
            model_config = asdict(model_config)
        if payload['phase'] != 'sft':
            raise ValueError('Require SFT initialization')
        tokenizer = root / 'data/tokenizer_8k.model'
        if sha(tokenizer) != payload['tokenizer_sha256']:
            raise ValueError('Tokenizer identity differs')
        manifest = read(data / 'filtered/manifest.json')
        for item in manifest['instructions']['files'].values():
            if sha(data / 'filtered' / item['path']) != item['sha256']:
                raise ValueError('Synthetic file identity differs')
        requests = [json.loads(line) for line in (data / 'requests.jsonl').read_text().splitlines()]
        train_requests = {row['id']: row for row in requests if row['split'] == 'train'}
        validation_requests = {row['id']: row for row in requests if row['split'] == 'validation'}
        if set(train_requests) & set(validation_requests) or len(validation_requests) != 120:
            raise ValueError('Unexpected request partition')
        train_rows = [json.loads(line) for line in (data / 'filtered/train.jsonl').read_text().splitlines()]
        if len(train_rows) != 391 or len({row['id'] for row in train_rows}) != len(train_rows):
            raise ValueError('Unexpected or duplicate accepted training examples')
        for row in train_rows:
            request = train_requests.get(row['id'])
            if (request is None or row['messages'][:-1] != request['messages']
                    or row['messages'][-1]['role'] != 'assistant'
                    or row['messages'][-1]['content'].strip() != request['expected']):
                raise ValueError('Training row is not an exact verified training request')
        families = dict(collections.Counter(row['family'] for row in train_rows))
        config = {'model': model_config, 'tokenizer': {'model_path': str(tokenizer)},
                  'finetune': dict(max_steps=100, warmup_steps=10, batch_size=8,
                      gradient_accumulation_steps=2, optimizer='adamw', learning_rate=5e-5,
                      muon_learning_rate=.001, min_lr_ratio=.1, decay_fraction=.8,
                      weight_decay=.01, grad_clip=1., eval_interval=50, eval_batches=8,
                      save_interval=100, log_interval=10, seed=20261013)}
        (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        evaluator = data / 'source/evaluate_mini_instructions.py'
        provenance = {str(path): sha(path) for path in [checkpoint, tokenizer, evaluator,
            data / 'requests.jsonl', data / 'filtered/train.jsonl', data / 'filtered/validation.jsonl',
            data / 'filtered/manifest.json', root / 'scripts/train_mini.py', root / 'src/model.py',
            Path(__file__), Path(__file__).with_name('train_verified_instructions.py')]}
        write(output / 'inputs.json', dict(hashes=provenance, initial_step=payload['step'],
              config=config, train_examples=len(train_rows), validation_requests=len(validation_requests),
              training_families=families,
              native_mode=args.native_mode, norm_mode=args.norm_mode,
              accepted_validation_examples=manifest['instructions']['files']['validation']['examples'],
              optimizer='new state, same for both backends',
              quality='last.pt after 100 updates; no selection by exact-match evaluation',
              limitations=['391 teacher-verified examples; no accepted JSON-family training examples',
                           '120 held-out input/paraphrase cases; original 60 are a separate skill diagnostic']))
        del payload
        env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONPATH=str(root))
        trainer = Path(__file__).with_name('train_verified_instructions.py')
        for label, mode in [('reference', 'reference'), ('native', args.native_mode)]:
            execute(label, [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=2', trainer, '--brandon-root', root, '--checkpoint', checkpoint,
                '--config', output / 'config.yaml', '--data-dir', data / 'filtered',
                '--output', output / label, '--mode', mode, '--norm-mode', args.norm_mode], env)
        for rank in (0, 1):
            if read(output / f'reference/batches-rank{rank}.json') != read(output / f'native/batches-rank{rank}.json'):
                raise RuntimeError('Reference and native consumed different batches')
        evaluations = {}
        for name, weights in [('before', checkpoint), ('reference', output / 'reference/last.pt'),
                              ('native', output / 'native/last.pt')]:
            evaluations[name] = {}
            for suite, maximum in [('transformations', 64), ('original', 32)]:
                destination = output / f'{name}-{suite}.json'
                command = [sys.executable, evaluator, '--checkpoint', weights, '--tokenizer', tokenizer,
                           '--output', destination, '--max-new-tokens', str(maximum)]
                if suite == 'transformations':
                    command += ['--cases', data / 'requests.jsonl']
                execute(f'{name}-{suite}', command, env)
                report = read(destination)
                evaluations[name][suite] = dict(correct=sum(case['correct'] for case in report['cases']),
                     total=len(report['cases']), scores=report['scores'],
                     stopped=sum(case['stopped'] for case in report['cases']),
                     exact_text=sum(case['response'].strip() == case['expected'] for case in report['cases']))
        write(output / 'results.json', dict(evaluations=evaluations, batch_hashes_match=True,
              reference=read(output / 'reference/summary.json'), native=read(output / 'native/summary.json')))
        audit = [sys.executable, Path(__file__).with_name('analyze_verified_instructions.py'),
                 '--output', output, '--source-checkpoint', checkpoint]
        if args.norm_mode == 'compatible' or args.native_mode == 'swiglu':
            audit.append('--require-exact')
        execute('numerical-audit', audit, env)
        update(state='complete', stage='done', reason='Paired 100-step SFT and all six evaluations completed')
    except Exception as error:
        update(state='failed', reason=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if lock is not None:
            lock.close()
        with tarfile.open(output / 'results.tar.gz', 'w:gz') as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.yaml', '.log', '.txt'):
                    archive.add(path, arcname=str(path.relative_to(output)))
        write(output / 'archive_ready.json', {'state': status['state']})
    return 0 if status['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
