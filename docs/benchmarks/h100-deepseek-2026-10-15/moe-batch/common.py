import contextlib, datetime, fcntl, json, os, resource, signal, subprocess, time
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-moe-batch-20261015')
host=root.parent
old=host/'ds41-tail-20261015'
prefix=host/'env'
source=root/'source'
model=host/'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',DS41_GPU='1',DS41_GPU_ATTN='1',DS41_VRAM_GB='62',DS41_PROFILE='1',DS41_GPU_TIMING='1')
env['PATH']=str(prefix/'bin')+':'+env['PATH']
env['LD_LIBRARY_PATH']=str(prefix/'targets/x86_64-linux/lib')+':'+str(prefix/'lib')
for k in ['DS41_MOE_BATCH','DS41_MOE_TRACE','DS41_GROUPED_TAIL','DS41_HC_DIRECT','DS41_CPU_GRAIN','DS41_MOE_OVERLAP','DS41_GPU_TEST','DS41_DUMP_STAGES','DS41_GPU_SELFTEST','DS41_GPU_SELFTEST2','DS41_MICROBENCH']:
    env.pop(k,None)
def status(state,**extra):
    (root/'status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
def run(label,args,extra=None,timeout=600):
    status('running',case=label)
    e=dict(env,**(extra or {}));cmd=list(map(str,args))
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (root/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    record=dict(command=cmd,returncode=code,seconds=time.monotonic()-start,
        environment={k:v for k,v in e.items() if k.startswith('DS41_') or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']},
        resources={k:getattr(after,k)-getattr(before,k) for k in ['ru_utime','ru_stime','ru_majflt','ru_minflt','ru_nvcsw','ru_nivcsw']})
    (root/(label+'-command.json')).write_text(json.dumps(record,indent=2))
    print(label,code,round(record['seconds'],2),flush=True)
    if code:raise RuntimeError(label+' failed: '+(root/(label+'.log')).read_text()[-4000:])
@contextlib.contextmanager
def admitted(label, wait_seconds=0):
    stage=Path('/lambda/nfs/btiny/brandon-mini/stages')
    deadline=time.monotonic()+wait_seconds
    while True:
        stack=contextlib.ExitStack()
        try:
            for path in sorted(list(stage.glob('*/pipeline.lock'))+[host/'exclusive-training.lock',host/'gpu1-correctness.lock']):
                f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def gate():
                states={p.parent.name:json.loads(p.read_text()) for p in stage.glob('*/status.json')}
                if any(d.get('state')!='complete' for d in states.values()):raise BlockingIOError('unfinished stage')
                if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise BlockingIOError('GPU busy')
                return states
            gate();time.sleep(2)
            (root/('admission-'+label+'.json')).write_text(json.dumps(gate(),indent=2))
            break
        except BlockingIOError as e:
            stack.close()
            if time.monotonic()>=deadline:raise
            status('waiting',reason=str(e));print('waiting for pipeline/GPU locks',flush=True)
            time.sleep(10)
        except Exception:
            stack.close();raise
    try:yield
    finally:stack.close()
def configure():
    run('configure',['cmake','-S',source,'-B',root/'build','-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-gcc'),'-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc')])
def compile_profile(label):
    run('compile-'+label,[prefix/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(source/'src'),'-I'+str(source/'include'),root/'profile.cpp',root/'build/libntransformer_lib.a',
        '-L'+str(prefix/'lib'),'-L'+str(prefix/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static',
        '-ldl','-lrt','-lpthread','-o',root/('profile-'+label)])
