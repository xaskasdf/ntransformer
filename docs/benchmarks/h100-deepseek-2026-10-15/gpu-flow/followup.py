import json
from pathlib import Path
import subprocess
import run as r

def main():
    state=json.loads((r.ROOT/'status.json').read_text())['state']
    if state not in ['initial-complete','complete']:raise RuntimeError('previous measurement job not complete')
    cases=json.loads((r.ROOT/'followup-cases.json').read_text())
    with r.b.admitted():
        with (r.ROOT/'followup-telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            try:
                for c in cases:r.case(c['name'],c['flags'],c.get('prompt',32),c.get('repeats',3))
            finally:monitor.terminate();monitor.wait(timeout=10)
    r.b.status('complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:r.b.status('failed',error=repr(e));raise
