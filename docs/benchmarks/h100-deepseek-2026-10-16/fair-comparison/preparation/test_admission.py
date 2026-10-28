"""CPU-only admission checks; no real GPU calls or training locks are used."""
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate',HERE/'admission-next.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def check(state,busy=False,held=False,allow=False):
    with tempfile.TemporaryDirectory(dir=HERE,prefix='admission-fixture-') as temporary:
        root=Path(temporary).resolve();assert root.is_relative_to(HERE.resolve())
        stages=root/'stages';stage=stages/'fixture';stage.mkdir(parents=True)
        (stage/'status.json').write_text(json.dumps(state));lock=stage/'pipeline.lock'
        snapshots=[];calls=[]
        b=SimpleNamespace(STAGES=stages,HOST=root,
            capture=lambda args: calls.append(args) or ('12345\n' if busy else ''),
            save=lambda name,value:snapshots.append(value),status=lambda *args,**kwargs:None)
        clock=iter([0,4001]);a.time.monotonic=lambda:next(clock);a.time.sleep=lambda _:None
        with contextlib.ExitStack() as stack:
            if held:
                stream=stack.enter_context(lock.open('a'));fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            admitted=False
            try:
                with a.admitted(b):admitted=True
            except BlockingIOError:pass
            assert admitted==allow,(state,busy,held,allow)
            if allow:assert len(calls)==2 and snapshots

if __name__=='__main__':
    check(dict(state='paused'),allow=True)
    check(dict(state='paused',pid=2147483647),allow=True)
    check(dict(state='paused',pid=os.getpid()))
    check(dict(state='paused'),busy=True)
    check(dict(state='paused'),held=True)
    check(dict(state='running'))
    check(dict(state='complete'),allow=True)
    check(dict(state='superseded'),allow=True)
    print('PASS 8 admission cases: paused idle/dead accepted; live PID, GPU workload, held lock and running stage remain blocked')
