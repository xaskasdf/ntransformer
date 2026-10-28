import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
deadline = time.monotonic()+1800
while True:
    state = json.loads((ROOT/'status.json').read_text())['state']
    if state == 'contexts-complete': break
    if state == 'failed' or time.monotonic() > deadline:
        raise RuntimeError('context suite did not complete: '+state)
    time.sleep(10)
subprocess.run([sys.executable, str(ROOT/'finalize.py')], check=True)
subprocess.run([sys.executable, str(ROOT/'seal.py')], check=True)
