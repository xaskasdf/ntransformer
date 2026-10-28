import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from contextlib import ExitStack
root=Path('/home/ubuntu/ntransformer-h100')
out=root/'training/rope-rounding-20261014-r2'
out.mkdir(exist_ok=False)
for name in ('probe-rope-rounding.cu','probe-rope-rounding.py','run-rope-rounding.py'):
    shutil.copy2(root/name,out/name)
env=dict(os.environ)
env['PATH']=str(root/'env/bin')+':'+env['PATH']
env['LD_LIBRARY_PATH']=f'{root}/env/targets/x86_64-linux/lib:{root}/env/lib:'+env.get('LD_LIBRARY_PATH','')
with (out/'build.log').open('w') as log:
    subprocess.run([str(root/'env/bin/nvcc'),'-std=c++20','-O3','--fmad=false','-arch=sm_90',
        '-ccbin',str(root/'env/bin/x86_64-conda-linux-gnu-gcc'),'-shared','-lstdc++','-Xcompiler=-fPIC',
        str(out/'probe-rope-rounding.cu'),'-o',str(out/'probe.so')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
stage=Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
deadline=time.monotonic()+7200
while True:
    stack=ExitStack()
    try:
        for path in (stage,stage.parent/'behavior-v1'):
            assert json.loads((path/'status.json').read_text())['state']=='complete'
            lock=stack.enter_context((path/'pipeline.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise BlockingIOError('GPU busy')
        break
    except BlockingIOError:
        stack.close()
        if time.monotonic()>deadline:raise RuntimeError('GPU admission timeout')
        time.sleep(15)
with stack, (out/'probe.log').open('w') as log:
    subprocess.run(['/home/ubuntu/brandon-mini/.venv/bin/python',str(out/'probe-rope-rounding.py')],
                   env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
print((out/'probe.log').read_text())
