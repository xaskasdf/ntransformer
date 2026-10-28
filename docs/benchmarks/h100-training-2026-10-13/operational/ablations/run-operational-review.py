import fcntl
import json
import os
import signal
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

root=Path('/home/ubuntu/ntransformer-h100')
stage=Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
out=root/'training/operational-ablation-20261013'
python='/home/ubuntu/brandon-mini/.venv/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
         NTRANSFORMER_TRAINING_LIB=str(root/'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH']=f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:'+env.get('LD_LIBRARY_PATH','')

def run(command, label):
    (out/'status.json').write_text(json.dumps(dict(state='running',stage=label)))
    with (out/(label+'.log')).open('x') as log:
        process=subprocess.Popen(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=600)
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

cases=[('bucket-views','native',['--deterministic','--skip-uninitialized-fill','--gradient-bucket-views']),
       ('deferred-readbacks','native',['--deterministic','--skip-uninitialized-fill','--defer-loss-readbacks']),
       ('combined','native',['--deterministic','--skip-uninitialized-fill','--gradient-bucket-views','--defer-loss-readbacks']),
       ('combined-repeat','native',['--deterministic','--skip-uninitialized-fill','--gradient-bucket-views','--defer-loss-readbacks']),
       ('combined-default','native',['--gradient-bucket-views','--defer-loss-readbacks']),
       ('combined-peer','peer',['--deterministic','--skip-uninitialized-fill','--gradient-bucket-views','--defer-loss-readbacks'])]
out.mkdir(exist_ok=False)
source=out/'source-snapshot'
shutil.copytree(root/'source/python',source/'python',ignore=shutil.ignore_patterns('__pycache__'))
(source/'scripts').mkdir()
for name in ('train_brandon_experiment.py','compare_training_experiment.py','analyze_verified_instructions.py'):
    shutil.copy2(root/'source/scripts'/name,source/'scripts'/name)
baseline=Path('docs/benchmarks/h100-training-2026-10-13/baseline-stage1-20261013-a')
shutil.copytree(root/'source'/baseline,source/baseline)
shutil.copy2(__file__,out/'run-operational-review.py')
parent=root/'training/operational-20261013'
deadline=time.monotonic()+3600
while True:
    state=json.loads((parent/'status.json').read_text())['state']
    if state=='complete':
        assert json.loads((parent/'native-no-fill-exact.json').read_text())['status']=='passed'
        break
    if state in ('failed','deferred') or time.monotonic()>deadline:
        (out/'status.json').write_text(json.dumps(dict(state='deferred',reason='Parent review not complete')))
        raise SystemExit('No GPU work: parent review not complete')
    (out/'status.json').write_text(json.dumps(dict(state='waiting',stage='parent-review',pid=os.getpid())))
    time.sleep(15)
deadline=time.monotonic()+1800
while True:
    stack=ExitStack()
    try:
        for path in (stage,stage.parent/'behavior-v1'):
            assert json.loads((path/'status.json').read_text())['state']=='complete'
            lock=stack.enter_context((path/'pipeline.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:
            raise BlockingIOError('GPU processes still active')
        break
    except BlockingIOError:
        stack.close()
        (out/'status.json').write_text(json.dumps(dict(state='waiting',stage='exclusive-gpu-locks',pid=os.getpid())))
        if time.monotonic()>deadline:
            (out/'status.json').write_text(json.dumps(dict(state='deferred',stage='exclusive-gpu-locks')))
            raise SystemExit('GPU gate still occupied after 30 minutes; no experiment launched')
        time.sleep(15)
with stack:
    (out/'cases.json').write_text(json.dumps(cases,indent=2))
    try:
        for label,mode,flags in cases:
            run([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 source/'scripts/train_brandon_experiment.py',
                 '--brandon-root',stage.parent.parent/'sources/stage1-v1','--checkpoint',stage/'pretrain/last.pt',
                 '--data-dir',stage/'data','--output',out/label,'--mode',mode,
                 '--steps','100','--warmup-steps','10',*flags],label)
            if '--deterministic' in flags:
                run([python,source/'scripts/compare_training_experiment.py',
                     '--reference',root/'training/difference-review-20261013/deterministic-reference',
                     '--candidate',out/label,'--source-checkpoint',stage/'pretrain/last.pt',
                     '--output',out/(label+'-exact.json'),'--require-exact'],label+'-audit')
        for label,mode,flags in [cases[2]]:
            run([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 source/'scripts/train_brandon_experiment.py',
                 '--brandon-root',stage.parent.parent/'sources/stage1-v1','--checkpoint',stage/'pretrain/last.pt',
                 '--data-dir',stage/'data','--output',out/('profile-'+label),'--mode',mode,
                 '--steps','13','--warmup-steps','10','--profile','--profile-start','11',*flags],
                'profile-'+label)
        (out/'status.json').write_text(json.dumps(dict(state='complete',stage='done')))
    except Exception as error:
        (out/'status.json').write_text(json.dumps(dict(state='failed',reason=str(error))))
        raise
