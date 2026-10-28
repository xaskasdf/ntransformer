import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

ROOT = Path('/home/ubuntu/ntransformer-h100/ds41-bench-20261015')
HOST = ROOT.parent
ENGINE = HOST / 'ds41-shared-20261015'
PREFIX = HOST / 'env'
MODEL = HOST / 'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
STAGES = Path('/lambda/nfs/btiny/brandon-mini/stages')
ENV = {k: v for k, v in os.environ.items() if not k.startswith('DS41_')}
ENV.update(CUDA_VISIBLE_DEVICES='1', DS41_GPU='1', DS41_GPU_ATTN='1', DS41_VRAM_GB='62',
           DS41_PROFILE='1', DS41_SHARED_GPU='1', DS41_MOE_BATCH='1', DS41_GROUPED_TAIL='1',
           DS41_HC_DIRECT='1', DS41_MOE_OVERLAP='1',
           LD_LIBRARY_PATH=str(PREFIX/'targets/x86_64-linux/lib')+':'+str(PREFIX/'lib'))
ENV['PATH'] = str(PREFIX/'bin') + ':' + ENV['PATH']

def save(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2)+'\n')

def status(state, **extra):
    save('status.json', dict(state=state, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))

def capture(args):
    return subprocess.check_output(list(map(str, args)), text=True, env=ENV)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def command(label, args, timeout=1200):
    args = list(map(str, args))
    status('running', case=label)
    started = time.monotonic()
    with (ROOT/(label+'.log')).open('w') as log:
        proc = subprocess.Popen(args, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    save(label+'-command.json', dict(command=args, returncode=code, wall_seconds=time.monotonic()-started,
         environment={k:v for k,v in ENV.items() if k.startswith('DS41_') or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']}))
    print(label, code, round(time.monotonic()-started, 3), flush=True)
    if code: raise RuntimeError(label+' failed; see '+label+'.log')

@contextlib.contextmanager
def admission():
    deadline = time.monotonic()+900
    while True:
        locks = contextlib.ExitStack()
        try:
            for path in sorted(list(STAGES.glob('*/pipeline.lock')) + [HOST/'exclusive-training.lock',HOST/'gpu1-correctness.lock']):
                stream = locks.enter_context(path.open('a'))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def check():
                states = {p.parent.name:json.loads(p.read_text()) for p in STAGES.glob('*/status.json')}
                for name, value in states.items():
                    if value.get('state') == 'complete': continue
                    # A terminal failed audit is not a running training job. Keep
                    # its status untouched and require its recorded PID to be gone.
                    pid = value.get('pid')
                    if value.get('state') == 'failed' and pid and not Path('/proc',str(pid)).exists(): continue
                    raise BlockingIOError('active stage '+name)
                apps = capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip()
                if apps: raise BlockingIOError('GPU busy: '+apps)
                return states
            check(); time.sleep(2); save('admission.json', check())
            break
        except BlockingIOError as error:
            locks.close()
            if time.monotonic() >= deadline: raise
            status('waiting', reason=str(error)); time.sleep(10)
        except BaseException:
            locks.close(); raise
    try: yield
    finally: locks.close()

def main():
    command('compile', [PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(ENGINE/'source/src'),'-I'+str(ENGINE/'source/include'),ROOT/'benchmark_ds41.cpp',
        ENGINE/'build/libntransformer_lib.a','-L'+str(PREFIX/'lib'),'-L'+str(PREFIX/'targets/x86_64-linux/lib'),
        '-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'benchmark_ds41'])
    environment = dict(gpu=capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv']),
        cpu=capture(['lscpu']), memory=capture(['free','-b']), topology=capture(['nvidia-smi','topo','-m']),
        compiler=capture([PREFIX/'bin/x86_64-conda-linux-gnu-g++','--version']), cuda=capture([PREFIX/'bin/nvcc','--version']),
        binary_sha256=sha(ROOT/'benchmark_ds41'), library_sha256=sha(ENGINE/'build/libntransformer_lib.a'),
        cli_sha256=sha(ENGINE/'build/ntransformer'), source_sha256=sha(ROOT/'benchmark_ds41.cpp'),
        model_shards=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(MODEL.parent.glob('*.gguf'))],
        source_origin=str(ENGINE/'source'), corpus_sha256=sha(ROOT/'corpus.txt'))
    save('environment.json', environment)
    # Preserve the exact engine snapshot already validated in the previous run.
    shutil.copy2(ENGINE/'source.tar.gz', ROOT/'engine-source.tar.gz') if (ENGINE/'source.tar.gz').exists() else shutil.copy2(ENGINE/'source.tar', ROOT/'engine-source.tar')
    with admission():
        with (ROOT/'telemetry.csv').open('w') as monitor:
            gpu = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                '--format=csv','--loop-ms=1000'], stdout=monitor, stderr=subprocess.STDOUT)
            stop = threading.Event()
            def observe():
                with (ROOT/'compute-processes.jsonl').open('w') as log:
                    while not stop.is_set():
                        apps = capture(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'])
                        groups = {}
                        for line in apps.splitlines():
                            try: groups[line.split(',')[0]] = os.getpgid(int(line.split(',')[0]))
                            except (ValueError, ProcessLookupError): pass
                        log.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), apps=apps, process_groups=groups))+'\n'); log.flush()
                        stop.wait(3)
            thread = threading.Thread(target=observe, daemon=True); thread.start()
            try:
                for n in [32,128,512]:
                    label = 'p'+str(n)+'-d64'
                    command(label, ['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'benchmark_ds41',
                        MODEL,ROOT/'corpus.txt',n,64,3,ROOT/label,label], timeout=1800)
            finally:
                stop.set(); thread.join(timeout=10)
                gpu.terminate(); gpu.wait(timeout=10)
    save('after.json', dict(gpu=capture(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv']),
                           compute_apps=capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])))
    status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        status('failed', error=repr(error)); raise
