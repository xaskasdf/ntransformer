import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def status(state, **extra):
    value = dict(state=state, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temporary = ROOT / 'queue-status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(ROOT / 'queue-status.json')
def verify():
    for name, digest in json.loads((ROOT / 'frozen-inputs.json').read_text()).items():
        if sha(ROOT / name) != digest: raise RuntimeError('queued input changed: ' + name)
def execute_phase(name):
    # The shared admission guard has a one-hour retry window. A complete
    # Brandon pipeline can last longer; retry only admission failures.
    spec = importlib.util.spec_from_file_location('cache_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    while True:
        verify(); status('waiting-or-running', phase=name)
        try: module.main(); return
        except BlockingIOError as error:
            status('waiting', phase=name, reason=str(error)); time.sleep(10)

def main():
    status('queued', phase='trace')
    execute_phase('trace')
    execute_phase('run')
    verify(); status('analyzing')
    subprocess.run([sys.executable, str(ROOT / 'analyze.py')], check=True, cwd=ROOT)
    import prepare as p
    p.b.status('complete')
    status('archiving')
    subprocess.run([sys.executable, str(ROOT / 'seal.py')], check=True, cwd=ROOT)
    status('complete', archive=json.loads((ROOT / 'archive.json').read_text()),
        persist='/lambda/nfs/btiny/ntransformer-training/deepseek-cache-20261015')
    shutil.copy2(ROOT / 'queue-status.json', '/lambda/nfs/btiny/ntransformer-training/deepseek-cache-20261015/queue-status.json')

if __name__ == '__main__':
    try: main()
    except BaseException as error: status('failed', error=repr(error)); raise
