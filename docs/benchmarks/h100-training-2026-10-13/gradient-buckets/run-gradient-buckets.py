"""Bounded, isolated continuation suite; freeze sources only after admission."""
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
out = root/'training/gradient-buckets-20261015-r2'
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
out.mkdir(exist_ok=False)
source = out/'source-snapshot'
env = dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
           NTRANSFORMER_TRAINING_LIB=str(out/'libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH','')

def status(state, **details):
    temporary = out/'status.tmp'
    temporary.write_text(json.dumps(dict(state=state,pid=os.getpid(),**details),indent=2)+'\n')
    temporary.replace(out/'status.json')

def gate():
    states = {p.parent.name:json.loads(p.read_text()) for p in stage.parent.glob('*/status.json')}
    pending = {name:d for name,d in states.items() if d.get('state') != 'complete'}
    if pending:
        raise BlockingIOError('Unfinished Brandon stages: '+','.join(sorted(pending)))
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise BlockingIOError('GPU processes still active: '+active)
    return states

deadline = time.monotonic()+7200
while True:
    stack = ExitStack()
    try:
        for path in (stage/'pipeline.lock',stage.parent/'behavior-v1/pipeline.lock',
                     stage.parent/'context-v2/pipeline.lock',root/'exclusive-training.lock'):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate()
        time.sleep(2)
        states = gate()
        break
    except BlockingIOError as error:
        stack.close()
        status('waiting',stage='admission',reason=str(error))
        if time.monotonic() > deadline:
            status('deferred',reason='Admission deadline exceeded; no GPU work launched')
            raise SystemExit(0)
        time.sleep(15)

def run(command,label):
    status('running',stage=label)
    (out/(label+'-command.json')).write_text(json.dumps(list(map(str,command)),indent=2)+'\n')
    with (out/(label+'.log')).open('x') as log:
        process = subprocess.Popen(list(map(str,command)),env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'{label}: exit {code}')
    print(label,flush=True)

base = ['--gradient-bucket-views','--defer-loss-readbacks','--shared-projection-casts',
        '--direct-gqa','--fused-rope','--loss-mode','fused','--fused-loss-labels','--training-graph']
strict = ['--deterministic','--skip-uninitialized-fill',*base]
variants = [('control',[]),('batched',['--batched-gradient-reduction']),
            ('buckets25',['--bucketed-graph']),
            ('buckets512',['--bucketed-graph','--gradient-bucket-mb','512'])]
cases=[]
for policy,flags in [('strict',strict),('normal',base)]:
    for suffix,ordered in [('',variants),('-repeat',list(reversed(variants)))]:
        cases.extend((policy+'-'+name+suffix,'native',flags+extra) for name,extra in ordered)
peer = [x for x in strict if x != '--training-graph']
cases += [('peer-control','peer',peer),('peer-batched','peer',peer+['--batched-gradient-reduction'])]

def train(label,mode,flags,steps=100,profile=False):
    run([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         source/'scripts/train_brandon_experiment.py',
         '--brandon-root',stage.parent.parent/'sources/stage1-v1',
         '--checkpoint',stage/'pretrain/last.pt','--data-dir',stage/'data',
         '--output',out/label,'--mode',mode,'--warmup-steps','0' if steps==2 else '10',
         '--steps',str(steps),*(['--profile','--profile-start','11'] if profile else []),*flags],label)

def audit(label,reference,exact):
    run([python,source/'scripts/compare_training_experiment.py','--reference',reference,
         '--candidate',out/label,'--source-checkpoint',stage/'pretrain/last.pt',
         '--output',out/(label+'-comparison.json'),*(['--require-exact'] if exact else [])],label+'-audit')

with stack:
    try:
        status('running',stage='freeze')
        (out/'admission.json').write_text(json.dumps(dict(states=states,scope='exclusive GPU benchmark',
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
        shutil.copy2(root/'build-training-labels/libntransformer_training.so',out/'libntransformer_training.so')
        shutil.copytree(root/'source/python',source/'python',ignore=shutil.ignore_patterns('__pycache__'))
        (source/'scripts').mkdir()
        for name in ('train_brandon_experiment.py','compare_training_experiment.py',
                     'analyze_verified_instructions.py','analyze_training_trace.py'):
            shutil.copy2(root/'source/scripts'/name,source/'scripts'/name)
        (source/'tests').mkdir()
        for name in ('test_training_loss.py','test_training_loss_labels.py','test_training_graph_storage.py','test_training_gradient_buckets.py'):
            shutil.copy2(root/'source/tests'/name,source/'tests'/name)
        shutil.copytree(root/'source/src/training',source/'src/training')
        (source/'src/cuda').mkdir()
        shutil.copy2(root/'source/src/cuda/rmsnorm.cu',source/'src/cuda/rmsnorm.cu')
        shutil.copy2(root/'source/CMakeLists.txt',source/'CMakeLists.txt')
        shutil.copytree(root/'training/labels-build-20261014',out/'build-record')
        shutil.copy2(root/'build-training-labels/compile_commands.json',out/'build-record/compile_commands.json')
        baseline = Path('docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a')
        shutil.copytree(root/'source'/baseline,source/baseline)
        shutil.copy2(__file__,out/'run-gradient-buckets.py')
        run([python,source/'tests/test_training_gradient_buckets.py',
             '--brandon-root',stage.parent.parent/'sources/stage1-v1',
             '--checkpoint',stage/'pretrain/last.pt','--data-dir',stage/'data',
             '--output',out/'gradient-bucket-tests.json'],'gradient-bucket-tests')
        for name,extra in variants:
            train('preflight-'+name,'native',strict+extra,steps=2)
            if name != 'control':audit('preflight-'+name,out/'preflight-control',True)
        (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        for label,mode,flags in cases:
            train(label,mode,flags)
            if '--deterministic' in flags:
                reference = root/'training/difference-review-20261013/deterministic-reference'
            elif label == 'normal-control':
                continue
            else:
                reference = out/'normal-control'
            audit(label,reference,'--deterministic' in flags)
        for label,mode,flags in cases[:4]:
            train('profile-'+label,mode,flags,steps=13,profile=True)
        status('complete',stage='done')
    except Exception as error:
        status('failed',reason=str(error))
        raise
