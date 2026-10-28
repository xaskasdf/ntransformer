import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

root=Path('/home/ubuntu/ntransformer-h100/ds41-prefill-20261015')
host=root.parent
stage=Path('/lambda/nfs/btiny/brandon-mini/stages')
source=root/'source'
source.mkdir(exist_ok=True)
with tarfile.open(root/'source.tar') as t: t.extractall(source,filter='data')
prefix=host/'env'
env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', DS41_GPU='1', DS41_VRAM_GB='62',
         DS41_GPU_ATTN='1', DS41_PROFILE='1', DS41_GPU_TIMING='1', PYTHONUNBUFFERED='1')
env['PATH']=str(prefix/'bin')+':'+env['PATH']
env['LD_LIBRARY_PATH']=str(prefix/'targets/x86_64-linux/lib')+':'+str(prefix/'lib')
for key in ['DS41_DUMP_STAGES','DS41_GPU_TEST','DS41_GPU_SELFTEST','DS41_GPU_SELFTEST2','DS41_MICROBENCH','DS41_PREFILL_ALL_LOGITS']:
    env.pop(key,None)

def status(state,**extra):
    (root/'status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')

def run(name,args,extra=None,timeout=600):
    case_env=dict(env,**(extra or {}))
    record=dict(command=list(map(str,args)),environment={k:case_env[k] for k in case_env
        if k.startswith('DS41_') or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']})
    status('running',case=name)
    start=time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        p=subprocess.Popen(record['command'],env=case_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
            raise
    record.update(returncode=code,seconds=time.monotonic()-start)
    (root/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(name,code,round(record['seconds'],3),flush=True)
    if code:
        print((root/(name+'.log')).read_text()[-5000:],flush=True)
        raise RuntimeError(name+' failed')

def gate():
    states={p.parent.name:json.loads(p.read_text()) for p in stage.glob('*/status.json')}
    if any(d.get('state')!='complete' for d in states.values()):
        raise BlockingIOError('unfinished Brandon stage')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
    if processes: raise BlockingIOError('GPU process active: '+processes)
    return states

try:
    run('configure',['cmake','-S',source,'-B',root/'build','-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-gcc'),
        '-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc')])
    run('build',['cmake','--build',root/'build','--target','ntransformer','test_ds41_prefill','test_ds41_backbone','-j','2'])
    with contextlib.ExitStack() as stack:
        for path in sorted(list(stage.glob('*/pipeline.lock'))+[host/'exclusive-training.lock',host/'gpu1-correctness.lock']):
            h=stack.enter_context(path.open('a'))
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gate(); time.sleep(2); states=gate()
        (root/'admission.json').write_text(json.dumps(states,indent=2)+'\n')
        shutil.copy2(host/'make_tiny_deepseek41.py',root/'make_tiny_deepseek41.original.py')
        run('backbone-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_backbone'])
        run('tiny-prefill',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_prefill$'],
            {'DS41_TEST_MODEL':'/home/ubuntu/tiny-deepseek41.gguf','DS41_GPU':'0'},timeout=300)
        model=host/'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
        for label,attn in [('cpu-attn','0'),('gpu-attn','1')]:
            run(label,[root/'build/ntransformer','-m',model,'--token-ids','0','--dump-logits',root/(label+'.bin'),'-n','0'],{'DS41_GPU_ATTN':attn})
        run('real-prefill',[root/'build/test_ds41_prefill',model,'16','4'],timeout=1200)
        for label,value in [('cli-all','1'),('cli-last','0')]:
            run(label,[root/'build/ntransformer','-m',model,'-p','The capital of France is','-n','4',
                '--temperature','0','--dump-logits',root/(label+'.bin')],{'DS41_PREFILL_ALL_LOGITS':value})
        status('complete')
except Exception as e:
    status('failed',error=repr(e))
    raise
