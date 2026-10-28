import fcntl
import json
import os
import subprocess
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
output = root / 'training/instructions-20261013'
pipeline = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1',
           NTRANSFORMER_TRAINING_LIB=str(root / 'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH', '')
with (pipeline / 'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('Another GPU workload is active')
    for name, mode in [('reference-repeat', 'reference'), ('swiglu-only', 'swiglu'), ('norms-only', 'norms')]:
        (output / 'audit-status.json').write_text(json.dumps({'state': 'running', 'stage': name}))
        command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   root / 'source/scripts/train_verified_instructions.py',
                   '--brandon-root', '/lambda/nfs/btiny/brandon-mini/sources/stage1-v1',
                   '--checkpoint', pipeline / 'sft/best.pt', '--config', output / 'config.yaml',
                   '--data-dir', '/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered',
                   '--output', output / name, '--mode', mode]
        with (output / 'logs' / f'audit-{name}.log').open('x') as log:
            subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        compare = [python, root / 'source/scripts/analyze_verified_instructions.py',
                   '--output', output, '--source-checkpoint', pipeline / 'sft/best.pt',
                   '--candidate', name, '--numerical-only']
        with (output / 'logs' / f'compare-{name}.log').open('x') as log:
            result = subprocess.run(list(map(str, compare)), env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'case': name, 'comparison_exit': result.returncode}), flush=True)
    (output / 'audit-status.json').write_text(json.dumps({'state': 'complete', 'stage': 'done'}))
