import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    # Replace only our old admission waiter; never signal a GPU workload.
    state=json.loads((ROOT/'status.json').read_text())
    previous=int(state['pid'])
    proc=Path('/proc',str(previous))
    if proc.exists():
        assert state['state']=='waiting', 'old worker is no longer an admission waiter'
        assert (proc/'cwd').resolve()==ROOT
        assert any((proc/'cmdline').read_bytes().endswith(name) for name in [b'trace.py\x00',b'queue.py\x00'])
        os.kill(previous,signal.SIGTERM)
        for _ in range(100):
            if not proc.exists():break
            time.sleep(.05)
        assert not proc.exists(), 'old admission waiter did not stop'
    environment=json.loads((ROOT/'environment.json').read_text())
    frozen={'source/'+name:digest for name,digest in environment['source_sha256'].items()}
    frozen.update({'bench':environment['binary_sha256'],'build/libntransformer_lib.a':environment['library_sha256']})
    for name in ['admission.py','prepare.py','trace.py','run.py','queue.py','analyze.py','seal.py','select_ds41_precision.py',
                 'replay_ds41_cache.cpp','corpus.txt','corpus-heldout.txt','environment.json','cpu-validation.json']:
        frozen[name]=sha(ROOT/name)
    for name,digest in frozen.items():assert sha(ROOT/name)==digest,name
    (ROOT/'frozen-inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
    with (ROOT/'queue.log').open('w') as output:
        process=subprocess.Popen([sys.executable,str(ROOT/'queue.py')],cwd=ROOT,
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,root=str(ROOT))),flush=True)

if __name__=='__main__':main()
