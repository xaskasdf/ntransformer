import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import common as b

ROOT=b.ROOT
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016')

def status(state,**extra):
    b.save('queue-status.json',dict(state=state,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))

def main():
    import run
    while True:
        b.verify();status('waiting-or-running',phase='pilot-and-comparison')
        try:run.main();break
        except BlockingIOError as error:
            status('waiting',reason=str(error));time.sleep(10)
    b.verify();status('analyzing')
    subprocess.run([b.PYTHON,ROOT/'analyze.py'],check=True,cwd=ROOT)
    b.status('complete');status('archiving')
    subprocess.run([b.PYTHON,ROOT/'seal.py'],check=True,cwd=ROOT)
    status('complete',archive=json.loads((ROOT/'archive.json').read_text()),persist=str(PERSIST))
    shutil.copy2(ROOT/'queue-status.json',PERSIST/'queue-status.json')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=repr(error));PERSIST.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'queue-status.json',PERSIST/'queue-status.json');raise
