import fcntl
import hashlib
import json
import os
import subprocess
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root / 'training/improve-20261013'
stage = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
brandon = Path('/lambda/nfs/btiny/brandon-mini/sources/stage1-v1')
old_sft = root / 'training/instructions-20261013'
behavior = Path('/lambda/nfs/btiny/brandon-mini/stages/behavior-v1')
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
    NTRANSFORMER_TRAINING_LIB=str(root / 'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH','')
def record(state, phase, **values):
    temporary = out / 'status.tmp'
    temporary.write_text(json.dumps(dict(state=state, stage=phase, **values), indent=2))
    temporary.replace(out / 'status.json')
def execute(name, command):
    record('running', name)
    with (out / (name + '.log')).open('x') as log:
        subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, flush=True)
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()
if (out / 'status.json').exists():
    raise RuntimeError('This run already has a status')
deadline = time.monotonic() + 3600
while True:
    if time.monotonic() > deadline:
        raise RuntimeError('Timed out waiting for the existing pipeline; no training launched')
    status = json.loads((behavior / 'status.json').read_text())
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if status.get('state') in ('complete', 'completed') and not processes:
        break
    record('waiting', 'behavior-v1', observed=status.get('stage'), gpu_pids=processes)
    time.sleep(10)
with ExitStack() as locks:
    for directory in (stage, behavior):
        lock = locks.enter_context((directory / 'pipeline.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(3):
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('A new GPU workload appeared; no training launched')
        time.sleep(2)
    try:
        checkpoint = stage / 'sft/best.pt'
        code = [root/'source'/p for p in ['src/training/ops.cu','python/ntransformer_training/ops.py',
            'python/ntransformer_training/brandon.py','scripts/train_verified_instructions.py',
            'scripts/train_brandon_experiment.py']]
        code += [Path(env['NTRANSFORMER_TRAINING_LIB']),checkpoint]
        (out/'inputs.json').write_text(json.dumps({'hashes':{str(p):sha(p) for p in code}},indent=2))
        execute('gpu-validation', [python,root/'source/tests/test_training_ops.py','--brandon-root',brandon,
            '--checkpoint',stage/'pretrain/last.pt','--norm-mode','compatible','--output',out/'validation.json'])
        execute('sft-compatible',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            root/'source/scripts/train_verified_instructions.py','--brandon-root',brandon,'--checkpoint',checkpoint,
            '--config',old_sft/'config.yaml','--data-dir','/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered',
            '--output',out/'sft-compatible','--mode','native','--norm-mode','compatible'])
        execute('compare-sft',[python,root/'source/scripts/analyze_verified_instructions.py','--output',out,
            '--source-checkpoint',checkpoint,'--reference',old_sft/'reference','--candidate','sft-compatible','--numerical-only'])
        comparison=json.loads((out/'comparison-sft-compatible.json').read_text())
        if any(comparison[key] != 0 for key in ['max_training_loss_delta','final_validation_delta','max_weight_difference']):
            raise AssertionError('Require bit-identical SFT, not just the earlier tolerance')
        summaries=[]
        for mode in ('reference','native','checkpoint','peer'):
            execute('pretrain-'+mode,[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                root/'source/scripts/train_brandon_experiment.py','--brandon-root',brandon,
                '--checkpoint',stage/'pretrain/last.pt','--data-dir',stage/'data','--output',out/mode,
                '--mode',mode,'--norm-mode','compatible','--steps','100','--warmup-steps','10'])
            summary=json.loads((out/mode/'summary.json').read_text())
            summaries.append(dict(summary,case=mode))
            if mode != 'reference':
                execute('compare-'+mode,[python,root/'source/scripts/compare_training_experiment.py',
                    '--reference',out/'reference','--candidate',out/mode,'--source-checkpoint',stage/'pretrain/last.pt',
                    '--output',out/('comparison-'+mode+'.json')])
        (out/'results.json').write_text(json.dumps({'cases':summaries},indent=2))
        record('complete','done')
    except Exception as error:
        record('failed','error',reason=str(error))
        raise
    finally:
        with tarfile.open(out/'results.tar.gz','w:gz') as archive:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix in ('.json','.jsonl','.log','.yaml','.txt'):
                    archive.add(p,arcname=str(p.relative_to(out)))
            for p in (root/'source/python/ntransformer_training').glob('*.py'):
                archive.add(p,arcname='source-snapshot/python/ntransformer_training/'+p.name)
            archive.add(root/'source/src/training/ops.cu',arcname='source-snapshot/src/training/ops.cu')
        (out/'archive_ready.json').write_text('{}')
