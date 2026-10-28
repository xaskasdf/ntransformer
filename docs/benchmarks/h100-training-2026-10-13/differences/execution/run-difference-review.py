import fcntl
import json
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root / 'training/difference-review-20261013'
stage = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
           NTRANSFORMER_TRAINING_LIB=str(root / 'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH', '')
with ExitStack() as stack:
    for pipeline in (stage, stage.parent / 'behavior-v1'):
        assert json.loads((pipeline / 'status.json').read_text())['state'] == 'complete'
        lock = stack.enter_context((pipeline / 'pipeline.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    out.mkdir(exist_ok=False)
    def run(label, command):
        (out / 'status.json').write_text(json.dumps(dict(state='running',stage=label)))
        with (out / (label + '.log')).open('w') as handle:
            subprocess.run(list(map(str, command)), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(label, flush=True)
    try:
        run('reference-repeat', [python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            root/'source/scripts/train_brandon_experiment.py', '--brandon-root',stage.parent.parent/'sources/stage1-v1',
            '--checkpoint',stage/'pretrain/last.pt','--data-dir',stage/'data','--output',out/'reference-repeat',
            '--mode','reference','--steps','100','--warmup-steps','10'])
        run('first-batch', [python,root/'source/scripts/probe_training_differences.py',
            '--brandon-root',stage.parent.parent/'sources/stage1-v1','--checkpoint',stage/'pretrain/last.pt',
            '--data-dir',stage/'data','--output',out/'first-batch.json'])
        (out/'status.json').write_text(json.dumps(dict(state='complete',stage='done')))
    except Exception as error:
        (out/'status.json').write_text(json.dumps(dict(state='failed',stage='error',reason=str(error))))
        raise
