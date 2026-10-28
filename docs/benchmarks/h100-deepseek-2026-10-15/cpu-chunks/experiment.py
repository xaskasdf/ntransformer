import contextlib, datetime, fcntl, hashlib, json, os, resource, shutil, signal, subprocess, tarfile, time
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-schedule-20261015')
host=root.parent
old=host/'ds41-prefill-20261015'
prefix=host/'env'
source=root/'source'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',DS41_GPU='1',DS41_GPU_ATTN='1',DS41_VRAM_GB='62',DS41_PROFILE='1')
env['PATH']=str(prefix/'bin')+':'+env['PATH']
env['LD_LIBRARY_PATH']=str(prefix/'targets/x86_64-linux/lib')+':'+str(prefix/'lib')
for k in ['DS41_CPU_GRAIN','DS41_GPU_TEST','DS41_DUMP_STAGES','DS41_GPU_SELFTEST','DS41_GPU_SELFTEST2','DS41_MICROBENCH']:
    env.pop(k,None)
def status(state,**extra):
    (root/'experiment-status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))
def run(label,args,extra=None,timeout=600):
    status('running',case=label)
    e=dict(env,**(extra or {}));cmd=list(map(str,args))
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (root/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=p.wait(timeout=timeout)
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
try:
    shutil.copytree(old/'source',source)
    with tarfile.open(root/'changes.tar') as t:t.extractall(source,filter='data')
    run('configure',['cmake','-S',source,'-B',root/'build','-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER='+str(prefix/'bin/x86_64-conda-linux-gnu-gcc'),'-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc')])
    run('build',['cmake','--build',root/'build','--target','ntransformer','test_ds41_parallel','test_ds41_prefill','-j','2'])
    run('compile-candidate',[prefix/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(source/'src'),'-I'+str(source/'include'),root/'profile.cpp',root/'build/libntransformer_lib.a',
        '-L'+str(prefix/'lib'),'-L'+str(prefix/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static',
        '-ldl','-lrt','-lpthread','-o',root/'profile-candidate'])
    run('parallel-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_parallel'],{'DS41_GPU':'0'})
    run('tiny-prefill',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_prefill$'],
        {'DS41_GPU':'0','DS41_TEST_MODEL':'/home/ubuntu/tiny-deepseek41.gguf'})
    stage=Path('/lambda/nfs/btiny/brandon-mini/stages')
    with contextlib.ExitStack() as stack:
        for path in sorted(list(stage.glob('*/pipeline.lock'))+[host/'exclusive-training.lock',host/'gpu1-correctness.lock']):
            f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def gate():
            states={p.parent.name:json.loads(p.read_text()) for p in stage.glob('*/status.json')}
            if any(d.get('state')!='complete' for d in states.values()):raise BlockingIOError('unfinished stage')
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise BlockingIOError('GPU busy')
            return states
        gate();time.sleep(2)
        (root/'admission-experiment.json').write_text(json.dumps(gate(),indent=2))
        model=host/'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
        # ABBA across processes; each first prompt+continuation is warm-up only.
        for label,binary in [('a1','profile-baseline'),('b1','profile-candidate'),('b2','profile-candidate'),('a2','profile-baseline')]:
            run(label,[root/binary,model,'16','3',root/(label+'.bin')])
        # Grain=1 restores the old dispatch in the new binary.
        run('control-grain1',[root/'profile-candidate',model,'16','1',root/'control-grain1.bin'],{'DS41_CPU_GRAIN':'1'})
        payloads=[(root/(label+'.bin')).read_bytes() for label in ['a1','b1','b2','a2','control-grain1']]
        if not all(x==payloads[0] for x in payloads):raise RuntimeError('cross-binary logits differ')
        (root/'exact.json').write_text(json.dumps(dict(bitwise_equal=True,logit_vectors=5,floats_per_vector=129280,
            sha256=hashlib.sha256(payloads[0]).hexdigest(),runs=['a1','b1','b2','a2','control-grain1']),indent=2))
    status('complete')
except Exception as e:
    status('failed',error=repr(e));raise
