import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
HOST = ROOT.parent
PREFIX = HOST / 'env'
STAGES = Path('/lambda/nfs/btiny/brandon-mini/stages')
MODEL = HOST / 'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
PYTHON = Path('/home/ubuntu/brandon-mini/.venv/bin/python')
CUBLAS = PYTHON.parent.parent / 'lib/python3.12/site-packages/nvidia/cu13'
ENV = {k:v for k,v in os.environ.items() if not k.startswith(('DS41_', 'GGML_', 'LLAMA_'))}
ENV.update(CUDA_VISIBLE_DEVICES='1,0', PATH=str(PREFIX/'bin')+':'+ENV['PATH'],
    LD_LIBRARY_PATH=':'.join(map(str, [ROOT/'llama-lib', PREFIX/'targets/x86_64-linux/lib', PREFIX/'lib', CUBLAS/'lib'])),
    GGML_BACKEND_PATH=str(ROOT/'llama-lib'), OMP_NUM_THREADS='52')
BASE_ENV = ENV.copy()
ABORT = threading.Event()

def save(name, value):
    path = ROOT/name
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)

def status(state, **extra):
    save('status.json', dict(state=state, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))

def capture(args): return subprocess.check_output(list(map(str,args)), env=ENV, text=True)

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def command(label, args, required=True, timeout=1800):
    args=list(map(str,args)); status('running', case=label)
    if (ROOT/(label+'-command.json')).exists(): raise RuntimeError('refusing to overwrite case '+label)
    start=time.monotonic(); utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/(label+'.log')).open('w') as log:
        process=subprocess.Popen(args, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline=time.monotonic()+timeout
            while True:
                if ABORT.is_set():raise RuntimeError('foreign GPU workload appeared; stopping our benchmark')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(args,timeout)
                try:code=process.wait(timeout=min(2,remaining));break
                except subprocess.TimeoutExpired:pass
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    save(label+'-command.json', dict(command=args, returncode=code, utc_start=utc,
        wall_seconds=time.monotonic()-start, process_group=process.pid,
        environment={k:v for k,v in ENV.items() if k.startswith(('DS41_','GGML_','LLAMA_','OMP_')) or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']}))
    print(label,code,round(time.monotonic()-start,3),flush=True)
    if code and required:raise RuntimeError(label+' failed')
    return code

def model_identity():
    return {p.name:dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in sorted(MODEL.parent.glob('*.gguf'))}

def verify():
    for name,digest in json.loads((ROOT/'frozen-inputs.json').read_text()).items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen input changed: '+name)
    if model_identity()!=json.loads((ROOT/'model-identity.json').read_text()):
        raise RuntimeError('model shard metadata changed')
    environment=json.loads((ROOT/'environment.json').read_text())
    for path,digest in environment.get('external_runtime_sha256',{}).items():
        if sha(path)!=digest:raise RuntimeError('resolved external runtime changed: '+path)
    if capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv'])!=environment['gpu']:
        raise RuntimeError('GPU/driver identity changed since preparation')
