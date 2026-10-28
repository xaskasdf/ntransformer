import contextlib, datetime, fcntl, json, os, signal, subprocess, time
from pathlib import Path
root = Path('/home/ubuntu/ntransformer-h100/ds41-schedule-20261015')
host = root.parent
old = host/'ds41-prefill-20261015'
prefix = host/'env'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', DS41_GPU='1', DS41_GPU_ATTN='1', DS41_VRAM_GB='62', DS41_PROFILE='1')
env['LD_LIBRARY_PATH'] = str(prefix/'targets/x86_64-linux/lib')+':'+str(prefix/'lib')
for k in ['DS41_GPU_TEST','DS41_DUMP_STAGES','DS41_GPU_SELFTEST','DS41_GPU_SELFTEST2','DS41_MICROBENCH']:
    env.pop(k,None)
def run(label, args, timeout=600):
    (root/'status.json').write_text(json.dumps(dict(state='running',case=label,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    with (root/(label+'.log')).open('w') as log:
        p = subprocess.Popen(list(map(str,args)), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try: code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
            raise
    if code: raise RuntimeError(label+' failed: '+(root/(label+'.log')).read_text()[-3000:])
    print(label,'complete',flush=True)
try:
    run('compile-probe',[prefix/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(old/'source/src'),'-I'+str(old/'source/include'),root/'profile.cpp',
        old/'build/libntransformer_lib.a','-L'+str(prefix/'lib'),'-L'+str(prefix/'targets/x86_64-linux/lib'),
        '-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',root/'profile-baseline'])
    stage=Path('/lambda/nfs/btiny/brandon-mini/stages')
    with contextlib.ExitStack() as stack:
        for path in sorted(list(stage.glob('*/pipeline.lock'))+[host/'exclusive-training.lock',host/'gpu1-correctness.lock']):
            f=stack.enter_context(path.open('a')); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def gate():
            states={p.parent.name:json.loads(p.read_text()) for p in stage.glob('*/status.json')}
            assert all(d.get('state')=='complete' for d in states.values()), states
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            return states
        gate();time.sleep(2)
        (root/'admission-probe.json').write_text(json.dumps(gate(),indent=2))
        run('baseline-profile',[root/'profile-baseline',host/'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf','16','3',root/'baseline-profile.bin'])
    (root/'status.json').write_text(json.dumps(dict(state='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
except Exception as e:
    (root/'status.json').write_text(json.dumps(dict(state='failed',error=repr(e))))
    raise
