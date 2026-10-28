import contextlib
import fcntl
import json
from pathlib import Path
import time

@contextlib.contextmanager
def admitted(b):
    deadline=time.monotonic()+3600
    while True:
        locks=contextlib.ExitStack()
        try:
            paths=sorted(list(b.STAGES.glob('*/pipeline.lock'))+
                [b.HOST/'exclusive-training.lock',b.HOST/'gpu1-correctness.lock'])
            for path in paths:
                handle=locks.enter_context(path.open('a'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def check():
                states={p.parent.name:json.loads(p.read_text()) for p in b.STAGES.glob('*/status.json')}
                for name,state in states.items():
                    if state.get('state')=='complete':continue
                    pids=[state.get(k) for k in ['pid','pipeline_pid','child_pid'] if state.get(k)]
                    if state.get('state')=='superseded' and not any(Path('/proc',str(pid)).exists() for pid in pids):continue
                    if state.get('state')=='failed' and pids and not any(Path('/proc',str(pid)).exists() for pid in pids):continue
                    raise BlockingIOError('active stage '+name)
                if b.capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
                    raise BlockingIOError('GPU busy')
                return states
            check();time.sleep(2);b.save('admission.json',check());break
        except BlockingIOError as error:
            locks.close()
            if time.monotonic()>deadline:raise
            b.status('waiting',reason=str(error));time.sleep(10)
        except BaseException:locks.close();raise
    try:yield
    finally:locks.close()
