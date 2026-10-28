#!/usr/bin/env python3
"""Wait for Brandon's entire pipeline, then validate and run bounded training A/Bs.

This supervisor never signals or changes another process. The original pipeline
lock remains held throughout the experiment to prevent a competing resume.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def gate_decision(status, summary, results_exists):
    state = status.get('state')
    if state == 'failed':
        return 'blocked', 'The original Brandon pipeline failed; no automatic experiment will start'
    if state != 'complete':
        return 'waiting', f'Original pipeline state={state}, stage={status.get("stage")}'
    if (not results_exists or not summary or summary.get('status') != 'complete'
            or summary.get('phase') != 'pretrain'
            or summary.get('completed_steps') != summary.get('planned_steps')
            or not isinstance(summary.get('planned_steps'), int) or summary['planned_steps'] < 1):
        return 'blocked', 'Completion metadata is inconsistent'
    return 'ready', 'Original pipeline completed; waiting for exclusive access and idle GPUs'


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                             '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=15)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--brandon-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--wait-hours', type=float, default=12)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if args.steps != 100 or args.poll_seconds < 1 or args.wait_hours <= 0:
        parser.error('The authorized experiment is 100 steps; polling/wait must be positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / 'supervisor.lock').open('a') as supervisor:
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'status.json').exists():
            raise RuntimeError('Experiment already has a status; use a new output directory')
        run(args, output)


def run(args, output):
    import fcntl
    root = Path(__file__).resolve().parents[1]
    pipeline = Path(args.pipeline).resolve()
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    state = dict(state='waiting', supervisor_pid=os.getpid(), pipeline=str(pipeline), steps=args.steps)

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(output / 'status.json', state)
        print(json.dumps(state), flush=True)

    def command(name, arguments, timeout=1800):
        update(state='running', stage=name, command=arguments)
        with (logs / f'{name}.log').open('w') as log:
            child = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=args.brandon_root, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only this experiment's newly created process group is signaled.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, arguments)

    stop_monitor = threading.Event()
    monitor_errors = []

    def monitor():
        with (output / 'gpu_processes.jsonl').open('a') as handle:
            while not stop_monitor.is_set():
                try:
                    record = dict(time=time.time(), stage=state.get('stage'), processes=gpu_processes())
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                except Exception as error:
                    monitor_errors.append(str(error))
                stop_monitor.wait(.5)

    lock = None
    worker = None
    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        update(reason='Waiting for the original pretrain, SFT and diagnostic pipeline')
        while True:
            status = json.loads((pipeline / 'status.json').read_text())
            summary_path = pipeline / 'pretrain/summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            decision, reason = gate_decision(status, summary, (pipeline / 'results.json').exists())
            if decision == 'blocked':
                update(state='blocked', reason=reason)
                return
            if decision == 'ready':
                lock = (pipeline / 'pipeline.lock').open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    lock = None
                if lock is not None:
                    # Re-read state after acquiring the lock; a resume may have won the race.
                    status = json.loads((pipeline / 'status.json').read_text())
                    if status['state'] == 'complete' and not gpu_processes():
                        # A short second observation catches just-starting external workloads.
                        time.sleep(2)
                        if not gpu_processes():
                            break
                    lock.close()
                    lock = None
            if time.monotonic() >= deadline:
                update(state='blocked', reason='Wait deadline reached without exclusive idle GPUs')
                return
            update(state='waiting', reason=reason, original_status=status)
            time.sleep(args.poll_seconds)

        checkpoint = pipeline / 'pretrain/last.pt'
        manifest_path = pipeline / 'data/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        update(state='running', stage='verify-inputs')
        for source in manifest['sources']:
            for item in source['files'].values():
                if sha256(pipeline / 'data' / item['path']) != item['sha256']:
                    raise RuntimeError(f'Corpus checksum mismatch: {item["path"]}')
        files = [root / 'CMakeLists.txt', root / 'src/cuda/rmsnorm.cu',
                 Path(os.environ['NTRANSFORMER_TRAINING_LIB'])]
        for folder, pattern in [('src/training', '*.cu'), ('python/ntransformer_training', '*.py')]:
            files.extend((root / folder).glob(pattern))
        files.extend(root / 'scripts' / name for name in ('train_brandon_experiment.py', 'queue_brandon_training.py'))
        files.extend(Path(args.brandon_root) / name for name in ('src/model.py', 'src/data/mini.py', 'scripts/train_mini.py'))
        atomic_json(output / 'inputs.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                    manifest_sha256=sha256(manifest_path), files={str(path): sha256(path) for path in files},
                    original_results=json.loads((pipeline / 'results.json').read_text()),
                    hypothesis='Native fused kernels and an event-ordered speculative peer activation pipeline'))
        worker = threading.Thread(target=monitor, daemon=True)
        worker.start()
        command('validate', [sys.executable, str(root / 'tests/test_training_ops.py'),
                '--brandon-root', args.brandon_root, '--checkpoint', str(checkpoint),
                '--output', str(output / 'validation.json')])

        def experiment(name, mode, devices, steps=100, profile=False):
            arguments = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={devices}',
                         str(root / 'scripts/train_brandon_experiment.py'), '--brandon-root', args.brandon_root,
                         '--checkpoint', str(checkpoint), '--data-dir', str(pipeline / 'data'),
                         '--output', str(output / name), '--mode', mode, '--steps', str(steps),
                         '--warmup-steps', str(min(10, steps - 1))]
            if profile:
                arguments.append('--profile')
            command(name, arguments)

        # Fail early on NCCL/DDP/recomputation before authorizing the 100-step series.
        experiment('preflight-reference', 'reference', 2, steps=2)
        experiment('preflight-peer', 'peer', 2, steps=2)
        command('compare-preflight', [sys.executable, str(root / 'scripts/compare_training_experiment.py'),
                '--reference', str(output / 'preflight-reference'), '--candidate', str(output / 'preflight-peer'),
                '--source-checkpoint', str(checkpoint), '--output', str(output / 'preflight-comparison.json')])
        cases = [('ddp-reference', 'reference', 2), ('ddp-native', 'native', 2),
                 ('ddp-checkpoint', 'checkpoint', 2), ('ddp-peer', 'peer', 2),
                 ('single-checkpoint', 'checkpoint', 1), ('single-peer', 'peer', 1)]
        summaries = []
        for name, mode, devices in cases:
            experiment(name, mode, devices)
            summary = json.loads((output / name / 'summary.json').read_text())
            summary['case'] = name
            if summaries and summary['batch_sha256'] != summaries[0]['batch_sha256']:
                raise RuntimeError('Compared runs used different sampled batches')
            summaries.append(summary)
            atomic_json(output / 'results.json', dict(status='running', cases=summaries))
        for name in ('ddp-native', 'ddp-checkpoint', 'ddp-peer', 'single-checkpoint', 'single-peer'):
            command(f'compare-{name}', [sys.executable, str(root / 'scripts/compare_training_experiment.py'),
                    '--reference', str(output / 'ddp-reference'), '--candidate', str(output / name),
                    '--source-checkpoint', str(checkpoint), '--output', str(output / f'comparison-{name}.json')])
        # Separate short traces: instrumented throughput is never part of the A/B table.
        experiment('profile-single-checkpoint', 'checkpoint', 1, steps=2, profile=True)
        experiment('profile-single-peer', 'peer', 1, steps=2, profile=True)
        atomic_json(output / 'results.json', dict(status='complete', cases=summaries, monitor_errors=monitor_errors))
        update(state='complete', stage='complete', reason='Validation and all bounded comparisons completed')
    except Exception as error:
        update(state='failed', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_monitor.set()
        if worker:
            worker.join(timeout=15)
        if lock:
            lock.close()
        # Small portable records only: model checkpoints and large traces stay remote.
        with tarfile.open(output / 'results.tar.gz.tmp', 'w:gz') as archive:
            for path in sorted(output.rglob('*')):
                if (path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.txt')
                        and not path.name.startswith('trace-')):
                    archive.add(path, arcname=str(path.relative_to(output)))
        (output / 'results.tar.gz.tmp').replace(output / 'results.tar.gz')
        atomic_json(output / 'archive_ready.json', dict(state=state['state'], archive='results.tar.gz'))


if __name__ == '__main__':
    main()
