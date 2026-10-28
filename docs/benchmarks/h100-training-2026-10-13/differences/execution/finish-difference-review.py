import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root/'training/difference-review-20261013'
stage = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ,OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
           NTRANSFORMER_TRAINING_LIB=str(root/'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH']=f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:'+env.get('LD_LIBRARY_PATH','')
deadline=time.monotonic()+1200
while json.loads((out/'status.json').read_text())['state']=='running':
    if time.monotonic()>deadline:
        raise TimeoutError('Deterministic comparison did not finish')
    time.sleep(5)
assert json.loads((out/'status.json').read_text())['state']=='complete'
with ExitStack() as stack:
    for pipeline in (stage,stage.parent/'behavior-v1'):
        lock=stack.enter_context((pipeline/'pipeline.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    def run(label,command):
        (out/'status.json').write_text(json.dumps(dict(state='running',stage=label)))
        with (out/(label+'.log')).open('x') as handle:
            subprocess.run(list(map(str,command)),env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print(label,flush=True)
    try:
        for mode in ('native','checkpoint','peer'):
            run('audit-'+mode,[python,root/'source/scripts/compare_training_experiment.py',
                '--reference',out/'deterministic-reference','--candidate',out/('deterministic-'+mode),
                '--source-checkpoint',stage/'pretrain/last.pt','--require-exact',
                '--output',out/('exact-'+mode+'.json')])
        for objective,checkpoint,data in (
            ('pretrain',stage/'pretrain/last.pt',stage/'data'),
            ('sft',stage/'sft/best.pt',stage.parent.parent/'distillation/text-v1/filtered')):
            run('probe-'+objective,[python,root/'source/scripts/probe_training_differences.py',
                '--objective',objective,'--brandon-root',stage.parent.parent/'sources/stage1-v1',
                '--checkpoint',checkpoint,'--data-dir',data,'--output',out/('probe-'+objective+'.json')])
        run('cpu-audit-tests',[python,root/'source/tests/test_training_audit.py'])
        (out/'status.json').write_text(json.dumps(dict(state='complete',stage='review-done')))
    except Exception as error:
        (out/'status.json').write_text(json.dumps(dict(state='failed',stage='review-error',reason=str(error))))
        raise
    finally:
        paths=[root/'source/scripts'/name for name in ('probe_training_differences.py','analyze_training_differences.py',
               'compare_training_experiment.py','train_brandon_experiment.py','analyze_verified_instructions.py')]
        paths += list((root/'source/python/ntransformer_training').glob('*.py'))
        paths += [root/'source/src/training/ops.cu',root/'source/tests/test_training_audit.py']
        hashes={}
        for path in paths:
            target=out/'source-snapshot'/path.relative_to(root/'source')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('run-difference-review.py','run-deterministic-review.py','finish-difference-review.py'):
            target=out/'execution'/name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(root/name,target)
        (out/'provenance.json').write_text(json.dumps(dict(source_sha256=hashes,
            gpu_processes_at_completion=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()),indent=2))
        with tarfile.open(out/'results.tar.gz','w:gz') as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.jsonl','.log','.py','.cu','.txt'):
                    archive.add(path,arcname=str(path.relative_to(out)))
        (out/'archive_ready.json').write_text('{}')
