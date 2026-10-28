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
out = root / 'training/target-loss-20261014'
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
           NTRANSFORMER_TRAINING_LIB=str(out/'libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH', '')
source = out/'source-snapshot'

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
        if code and label == 'strict-stream-audit' and (out/'strict-stream-comparison.json').exists():
            print('Experimental stream audit rejected; retaining evidence', flush=True)
            return
        if code:
            raise RuntimeError(f'{label}: exit {code}')
    print(label, flush=True)

combined = ['--gradient-bucket-views', '--defer-loss-readbacks', '--shared-projection-casts', '--direct-gqa', '--fused-rope']
strict = ['--deterministic', '--skip-uninitialized-fill', *combined]
shared = ['--loss-mode', 'fused']
cases = [('normal-fused-repeat', 'native', combined+shared),
         ('normal-control-repeat', 'native', combined)]

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
        original=json.loads((out/'cases.json').read_text())
        assert not any(label in [x[0] for x in original] for label,_,_ in cases)
        for label,mode,flags in cases:
            train(label,mode,flags)
        run([python,source/'scripts/compare_training_experiment.py',
             '--reference',out/'normal-control-repeat','--candidate',out/'normal-fused-repeat',
             '--source-checkpoint',stage/'pretrain/last.pt',
             '--output',out/'normal-fused-repeat-comparison.json'],'normal-fused-repeat-audit')
        run([python,source/'scripts/compare_training_experiment.py',
             '--reference',out/'normal-control','--candidate',out/'normal-control-repeat',
             '--source-checkpoint',stage/'pretrain/last.pt',
             '--output',out/'normal-control-repeat-comparison.json'],'normal-control-repeat-audit')
        (out/'cases.json').write_text(json.dumps(original+cases,indent=2)+'\n')
        shutil.copy2(__file__,out/'repeat-target-loss.py')
        status('complete',stage='done-with-normal-repeats')
    except Exception as error:
        status('failed',reason=str(error))
        raise
