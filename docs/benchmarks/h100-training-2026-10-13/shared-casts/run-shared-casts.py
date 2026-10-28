import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
stage = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
out = root / 'training/shared-casts-20261014-r2'
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
           NTRANSFORMER_TRAINING_LIB=str(root/'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH', '')
out.mkdir(exist_ok=False)
source = out / 'source-snapshot'
shutil.copytree(root/'source/python', source/'python', ignore=shutil.ignore_patterns('__pycache__'))
(source/'scripts').mkdir()
for name in ('train_brandon_experiment.py', 'compare_training_experiment.py',
             'analyze_verified_instructions.py', 'analyze_training_trace.py'):
    shutil.copy2(root/'source/scripts'/name, source/'scripts'/name)
(source/'tests').mkdir()
shutil.copy2(root/'source/tests/test_training_projection_casts.py', source/'tests')
baseline = Path('docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a')
shutil.copytree(root/'source'/baseline, source/baseline)
shutil.copy2(__file__, out/'run-shared-casts.py')

def status(state, **kwargs):
    (out/'status.json').write_text(json.dumps(dict(state=state, pid=os.getpid(), **kwargs)))

def run(command, label):
    status('running', stage=label)
    with (out/(label+'.log')).open('x') as log:
        process = subprocess.Popen(list(map(str, command)), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f'{label}: exit {code}')
    print(label, flush=True)

combined = ['--gradient-bucket-views', '--defer-loss-readbacks']
strict = ['--deterministic', '--skip-uninitialized-fill', *combined]
shared = ['--shared-projection-casts']
cases = [('strict-control', 'native', strict),
         ('strict-shared', 'native', strict+shared),
         ('normal-control', 'native', combined),
         ('normal-shared', 'native', combined+shared),
         ('strict-shared-repeat', 'native', strict+shared),
         ('strict-control-repeat', 'native', strict),
         ('peer-control', 'peer', strict),
         ('peer-shared', 'peer', strict+shared)]

def train(label, mode, flags, profile=False):
    run([python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
         source/'scripts/train_brandon_experiment.py',
         '--brandon-root', stage.parent.parent/'sources/stage1-v1',
         '--checkpoint', stage/'pretrain/last.pt', '--data-dir', stage/'data',
         '--output', out/label, '--mode', mode, '--warmup-steps', '10',
         '--steps', '13' if profile else '100',
         *(['--profile', '--profile-start', '11'] if profile else []), *flags], label)

deadline = time.monotonic()+7200
while True:
    stack = ExitStack()
    try:
        for path in (stage, stage.parent/'behavior-v1'):
            assert json.loads((path/'status.json').read_text())['state'] == 'complete'
            lock = stack.enter_context((path/'pipeline.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise BlockingIOError('GPU processes still active')
        break
    except BlockingIOError:
        stack.close()
        status('waiting', stage='exclusive-gpu-locks')
        if time.monotonic() > deadline:
            status('deferred', stage='exclusive-gpu-locks')
            raise SystemExit('No GPU work: admission deadline exceeded')
        time.sleep(15)
with stack:
    try:
        run([python, source/'tests/test_training_projection_casts.py',
             '--brandon-root', stage.parent.parent/'sources/stage1-v1',
             '--checkpoint', stage/'pretrain/last.pt', '--data-dir', stage/'data',
             '--output', out/'projection-tests.json'], 'projection-tests')
        (out/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
        for label, mode, flags in cases:
            train(label, mode, flags)
            if '--deterministic' in flags:
                reference = root/'training/difference-review-20261013/deterministic-reference'
            elif label == 'normal-control':
                continue
            else:
                reference = out/'normal-control'
            run([python, source/'scripts/compare_training_experiment.py',
                 '--reference', reference, '--candidate', out/label,
                 '--source-checkpoint', stage/'pretrain/last.pt',
                 '--output', out/(label+'-comparison.json'),
                 *(['--require-exact'] if '--deterministic' in flags else [])], label+'-audit')
        for label, mode, flags in cases[:2]:
            train('profile-'+label, mode, flags, profile=True)
        status('complete', stage='done')
    except Exception as error:
        status('failed', reason=str(error))
        raise
