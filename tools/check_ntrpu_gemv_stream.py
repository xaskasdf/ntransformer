"""Check complete GEMV transport/MAC simulation logs, not Vivado exit alone."""
import argparse
import json
from pathlib import Path
import re

# rows, columns, stimulus scenario, expected completion status
CASES = [(32,32,0,0),(3,2048,0,0),(2,8192,0,0),(32,1024,0,0),
         (2,2048,1,3),(2,2048,2,4),(2,2048,3,6),(1,1024,4,1),
         (1,1024,5,2),(1,1024,6,2),(1,1024,7,1),(1,1024,8,1),
         (2,1024,9,7),(1,1024,10,8),(1,1024,11,3),(1,1024,12,5),
         (1,1024,13,4),(3,2048,14,0),(2,2048,15,0),
         (1,1024,20,1),(1,1024,21,1),(1,1024,0,0)]

def review(text, seed, include_reset=True, lanes=1, extended_dual=True):
    errors=[]
    def need(condition,message):
        if not condition: errors.append(message)
    executed='\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    need(not re.search(r'^\s*(?:Fatal:|FATAL_ERROR|ERROR:|FAIL\b)',executed,re.M|re.I),'Executed error/fatal')
    records=[]
    for line in executed.splitlines():
        if line.startswith('NTRPU_GEMV_CASE '):
            fields=dict(re.findall(r'(\w+)=(\d+)',line))
            need(set(fields)=={'id','rows','columns','scenario','status','cycles','first_result','bursts','outputs'},'Malformed case')
            records.append({k:int(v) for k,v in fields.items()})
    contracts=CASES+([(1,1024,22,1)] if lanes==2 else [])
    if lanes==2 and extended_dual:
        contracts += [(2,2048,23,6),(2,2048,24,0),(1,1024,0,0)]
    need(lanes in {1,2},'Unsupported lane count')
    need([r.get('id') for r in records]==list(range(1,len(contracts)+1)),'Missing, duplicated or unordered cases')
    if lanes==2:
        need(re.findall(r'^NTRPU_GEMV_VARIANT lanes=(\d+) layout=(\d+)\s*$',executed,re.M)==[('2','2')],'Dual layout/variant missing')
    for r,want in zip(records,contracts):
        need(tuple(r.get(k) for k in ['rows','columns','scenario','status'])==want,'Case contract mismatch '+str(r.get('id')))
        n,_,scenario,status=want
        need(r.get('outputs')==(n if status==0 else 0),'Partial/missing output '+str(r.get('id')))
        need(0<r.get('cycles',0)<60000,'Cycle bound')
        first=r.get('first_result',0)
        need((0<first<r.get('cycles',0)) if status==0 else first==0,'Publication timing')
        need(r.get('bursts',0)==0 if scenario in {4,5,6,7,8,20,21,22} else r.get('bursts',0)>0,'AXI scope '+str(r.get('id')))
    footers=re.findall(r'^NTRPU_GEMV_STREAM_PASS (.+)$',executed,re.M)
    need(len(footers)==1,'Missing or duplicated suite marker')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',footers[0])} if footers else {}
    resets=3 if lanes==2 and extended_dual else 2
    need(summary.get('cases')==len(contracts)+(resets if include_reset else 0),'Suite case count')
    need(summary.get('result_stalls',0)>0 and summary.get('completion_stalls',0)>0,'Backpressure missing')
    minimum=sum(n*columns//32 for n,columns,_,status in contracts if status==0)
    need(summary.get('mac_ops',0)>=(2441 if lanes==1 else minimum),'MAC work missing')
    expected_bursts=sum(r.get('bursts',0) for r in records)+(resets*2 if include_reset else 0)
    need(summary.get('bursts')==expected_bursts,'Burst totals differ')
    need(summary.get('cycles',0)>=sum(r.get('cycles',0) for r in records),'Total cycles inconsistent')
    if include_reset:
        need(re.findall(r'^NTRPU_GEMV_SEED seed=(\d+)\s*$',executed,re.M)==[str(seed)],'Actual seed mismatch')
        need(re.findall(r'^NTRPU_GEMV_RESET_PASS phase=(AXI|MAC|BOTH) outputs=0\s*$',executed,re.M)==(['AXI','MAC','BOTH'] if resets==3 else ['AXI','MAC']),'Reset coverage missing')
    if lanes==2 and extended_dual:
        need(re.findall(r'^NTRPU_GEMV_NONASSOCIATIVE_PASS rows=2 outputs=2 value=0 (regrouped=1)[ \t]*$',executed,re.M)==['regrouped=1'],'Non-associative tile-boundary check missing')
    return {'status':'FAIL' if errors else 'PASS','errors':errors,'seed':seed,
            'cases':records,'summary':summary,'coordinated_reset_tested':include_reset,
            'scope':'Synthetic AXI memory and generated HLS MAC; no MIG/PCIe/physical throughput or model inference'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--legacy-22',action='store_true');p.add_argument('--json',type=Path)
    p.add_argument('--lanes',type=int,choices=[1,2],default=1)
    p.add_argument('--legacy-dual-25',action='store_true')
    a=p.parse_args();r=review(a.log.read_text(errors='replace'),a.seed,not a.legacy_22,a.lanes,not a.legacy_dual_25)
    if a.json:
        with a.json.open('x',encoding='utf8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r));return 0 if r['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
