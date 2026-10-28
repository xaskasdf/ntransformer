import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[1]/'docs/benchmarks/h100-deepseek-2026-10-16/fair-comparison'
HOST='ubuntu@<h100-host>'
REMOTE='/home/ubuntu/ntransformer-h100/ds41-fair-20261016'

def save(value):
    value['checked_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary=DATA/'queue-status.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    temporary.replace(DATA/'queue-status.json')

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,'cat '+REMOTE+'/queue-status.json'],
                capture_output=True,text=True,check=True,timeout=25)
            state=json.loads(result.stdout);save(state)
            if state['state']=='failed':
                subprocess.run(['scp','-q',HOST+':'+REMOTE+'/queue.log',HOST+':'+REMOTE+'/status.json',str(DATA)],check=True,timeout=60)
                (DATA/'STATUS.md').write_text('# Benchmark stopped\n\nThe worker reported a failure. See queue-status.json and queue.log. No new performance comparison is accepted.\n',encoding='utf-8',newline='\n')
                return
            if state['state']=='complete':
                subprocess.run([sys.executable,str(HERE/'fetch.py')],check=True,timeout=2100)
                subprocess.run([sys.executable,str(HERE/'publish_docs.py')],check=True,timeout=180)
                save(state|dict(local_collection='complete'))
                print('Completed, locally verified and documented.',flush=True);return
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            save(dict(state='collector-retrying',error=str(error)))
            print('Retrying:',repr(error),flush=True)
        time.sleep(30)

if __name__=='__main__':main()
