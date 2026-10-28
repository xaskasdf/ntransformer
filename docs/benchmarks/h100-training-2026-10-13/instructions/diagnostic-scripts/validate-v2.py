import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
output = root / 'training/instructions-20261013'
performance = root / 'training/precise-20261013'
pipeline = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
brandon = Path('/lambda/nfs/btiny/brandon-mini/sources/stage1-v1')
python = '/home/ubuntu/brandon-mini/.venv/bin/python'
env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0,1', PYTHONPATH=str(brandon),
           NTRANSFORMER_TRAINING_LIB=str(root / 'build-training/libntransformer_training.so'))
env['LD_LIBRARY_PATH'] = f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:' + env.get('LD_LIBRARY_PATH', '')
performance.mkdir(exist_ok=False)
def execute(name, command, required=True):
    (performance / 'status.json').write_text(json.dumps({'state': 'running', 'stage': name}))
    with (performance / f'{name}.log').open('x') as log:
        result = subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'stage': name, 'exit': result.returncode}), flush=True)
    if required and result.returncode:
        raise RuntimeError(name)
with (pipeline / 'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('Another GPU workload is active')
    paths = [root / 'source/CMakeLists.txt', root / 'source/src/training/ops.cu',
             root / 'source/python/ntransformer_training/brandon.py', root / 'build-training/libntransformer_training.so']
    (performance / 'inputs.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2))
    execute('validation', [python, root / 'source/tests/test_training_ops.py', '--brandon-root', brandon,
            '--checkpoint', pipeline / 'pretrain/last.pt', '--output', performance / 'validation.json'])
    for name, mode in [('swiglu-fixed', 'swiglu'), ('native-fixed', 'native')]:
        execute(name, [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                root / 'source/scripts/train_verified_instructions.py', '--brandon-root', brandon,
                '--checkpoint', pipeline / 'sft/best.pt', '--config', output / 'config.yaml',
                '--data-dir', '/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered',
                '--output', output / name, '--mode', mode])
        execute('compare-' + name, [python, root / 'source/scripts/analyze_verified_instructions.py',
                '--output', output, '--source-checkpoint', pipeline / 'sft/best.pt',
                '--candidate', name, '--numerical-only'], required=name == 'swiglu-fixed')
    for mode in ('reference', 'native'):
        execute('pretrain-' + mode, [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                root / 'source/scripts/train_brandon_experiment.py', '--brandon-root', brandon,
                '--checkpoint', pipeline / 'pretrain/last.pt', '--data-dir', pipeline / 'data',
                '--output', performance / mode, '--mode', mode, '--steps', '100', '--warmup-steps', '10'])
    execute('compare-pretrain', [python, root / 'source/scripts/compare_training_experiment.py',
            '--reference', performance / 'reference', '--candidate', performance / 'native',
            '--source-checkpoint', pipeline / 'pretrain/last.pt', '--output', performance / 'comparison.json'])
    for suite, maximum in [('transformations', 64), ('original', 32)]:
        command = [python, '/lambda/nfs/btiny/brandon-mini/distillation/text-v1/source/evaluate_mini_instructions.py',
                   '--checkpoint', output / 'swiglu-fixed/last.pt', '--tokenizer', brandon / 'data/tokenizer_8k.model',
                   '--output', output / f'swiglu-fixed-{suite}.json', '--max-new-tokens', str(maximum)]
        if suite == 'transformations':
            command += ['--cases', '/lambda/nfs/btiny/brandon-mini/distillation/text-v1/requests.jsonl']
        execute('evaluate-' + suite, command)
    (performance / 'status.json').write_text(json.dumps({'state': 'complete', 'stage': 'done'}))
