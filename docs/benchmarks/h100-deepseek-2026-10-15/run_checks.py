import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/home/ubuntu/ntransformer-h100/ds41-audit-20261015')
evidence = root / 'evidence'
evidence.mkdir(exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', DS41_GPU='1')
env['LD_LIBRARY_PATH'] = '/home/ubuntu/ntransformer-h100/env/targets/x86_64-linux/lib:/home/ubuntu/ntransformer-h100/env/lib'
for name in ['gputime.log', 'real_gpu_gen.log', 'real_prof.log', 'our_real0_gpu.log', 'microbench.log', 'real_final.log']:
    p = Path('/tmp') / name
    if p.is_file():
        shutil.copy2(p, evidence / name)
for name in ['run_gputime.sh', 'run_gpu_gen.sh', 'run_attn_verify.sh', 'run_final_bench.sh']:
    p = Path('/home/ubuntu') / name
    if p.is_file():
        shutil.copy2(p, evidence / name)

def query(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

result = dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=[],
    gpu_before=query(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv']),
    processes_before=query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv']),
    reference_commit=query(['git', '-C', '/home/ubuntu/llama-ds41', 'rev-parse', 'HEAD']))
with open('/home/ubuntu/ntransformer-h100/gpu1-correctness.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for mode in ['prologue', 'tail', 'compressor', 'indexer']:
        args = [str(root / 'test_ds41_backbone'), mode]
        t0 = time.monotonic()
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
        (root / (mode + '.log')).write_text(p.stdout + p.stderr)
        result['cases'].append(dict(command=args, returncode=p.returncode,
            seconds=time.monotonic()-t0, stdout=p.stdout, stderr=p.stderr))
    p = subprocess.run([str(root / 'test_ds41_backbone'), 'prologue'],
        env=dict(env, DS41_GPU='0'), capture_output=True, text=True, timeout=15)
    result['disabled'] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

result['gpu_after'] = query(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'])
result['ok'] = all(c['returncode'] == 0 for c in result['cases']) and result['disabled']['returncode'] == 77
result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in list((root/'source').rglob('*'))+[root/'test_ds41_backbone.cpp', root/'test_ds41_backbone'] if p.is_file()}
(root/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['ok'] else 1)
