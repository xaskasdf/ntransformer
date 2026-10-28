"""Validate activation-loader case contracts and drain/coverage counts."""
import argparse
import json
from pathlib import Path
import re

# blocks, offset, scenario, status, bursts, returned, outputs
CASES = [
    (1,0,0,0,1,1,1),(23,4032,0,0,2,23,23),(256,4032,0,0,5,256,256),
    (64,0,0,0,1,64,64),(33,64,0,0,1,33,33),
    (0,0,1,1,0,0,0),(257,0,1,1,0,0,0),(2,0,2,1,0,0,0),
    (2,0,3,1,0,0,0),(2,0,4,2,0,0,0),(2,0,5,1,0,0,0),(2,0,6,1,0,0,0),
    (23,4032,7,4,2,23,22),(23,4032,8,4,2,23,22),
    (23,0,9,4,1,23,0),(23,0,10,4,1,23,0),(130,4032,11,3,4,130,129),
    (64,0,12,3,1,64,2),(64,0,13,3,1,3,2),(128,0,14,7,0,0,0),
    (128,0,15,7,1,64,0),(128,0,16,7,1,64,3),(128,0,17,8,1,64,0),
    (128,0,18,7,1,64,1),(130,4032,20,4,4,130,129),
    (1,0,23,0,1,1,1),(1,0,24,0,1,1,1),(1,0,25,0,1,1,1),(1,0,26,0,1,1,1),
    (3,4032,0,0,2,3,3)]


def review(text, seed):
    errors=[]
    def need(ok,message):
        if not ok: errors.append(message)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(re.findall(r'^NTRPU_ACTIVATION_SEED seed=(\d+)\s*$',text,re.M)==[str(seed)],'Seed')
    records=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
             for line in text.splitlines() if line.startswith('NTRPU_ACTIVATION_CASE ')]
    need([r.get('id') for r in records]==list(range(1,26))+list(range(28,33)),'Case identities')
    for r,want in zip(records,CASES):
        need(set(r)=={'id','blocks','offset','scenario','status','cycles','bursts','returned','outputs'},'Case fields')
        need(tuple(r.get(k) for k in ['blocks','offset','scenario','status','bursts','returned','outputs'])==want,'Case contract '+str(r.get('id')))
        need(0<r.get('cycles',0)<6000,'Cycle bound')
    need(re.findall(r'^NTRPU_ACTIVATION_RESET case=(\d+) scenario=(\d+) outputs=(\d+)\s*$',text,re.M)==[('26','21','0'),('27','22','0')],'Reset coverage')
    footers=re.findall(r'^NTRPU_ACTIVATION_PASS (.+)$',text,re.M)
    need(len(footers)==1,'Suite marker')
    footer={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',footers[0])} if footers else {}
    need(set(footer)=={'cases','bursts','outputs','ar_stalls','output_stalls','done_stalls','resets'},'Suite fields')
    need(footer.get('cases')==32 and footer.get('resets')==2,'Suite cases/resets')
    need(footer.get('bursts')==sum(r.get('bursts',0) for r in records)+2,'Burst sum including resets')
    need(footer.get('outputs')==sum(r.get('outputs',0) for r in records),'Output sum')
    for key in ['ar_stalls','output_stalls','done_stalls']: need(footer.get(key,0)>0,'Missing '+key)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=records,summary=footer)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();result=review(a.log.read_text(),a.seed);print(json.dumps(result));return result['status']!='PASS'


if __name__=='__main__': raise SystemExit(main())
