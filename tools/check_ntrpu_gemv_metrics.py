"""Cross-check nonintrusive per-job cycle/handshake counters in GEMV RTL logs."""
import argparse,json,re
from pathlib import Path

PHASES=['idle','prepare','descriptor','run','results','complete']
FIELDS={'id',*PHASES,'lane0_busy','lane1_busy','both_busy','q8_stall','ar_stall',
        'r_stall','r_wait','r_beats','block_ops','activation_writes'}

def review(text,lanes):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    cases=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',l))
           for l in text.splitlines() if l.startswith('NTRPU_GEMV_CASE ')]
    metrics=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',l))
             for l in text.splitlines() if l.startswith('NTRPU_GEMV_METRIC ')]
    need(lanes in {1,2},'Invalid lanes')
    need(bool(cases) and [m.get('id') for m in metrics]==[c['id'] for c in cases],'Missing/order/duplicate metrics')
    totals={k:0 for k in FIELDS-{'id'}};cycles=0;successful=0
    for c,m in zip(cases,metrics):
        need(set(m)==FIELDS,'Metric fields')
        need(sum(m.get(k,0) for k in PHASES)==c['cycles'],'Cycle partition '+str(c['id']))
        for key in FIELDS-{'id','block_ops','activation_writes'}:
            need(0<=m.get(key,0)<=c['cycles'],'Counter bound '+key)
        busy=m.get('lane0_busy',0)+m.get('lane1_busy',0)-m.get('both_busy',0)
        need(0<=busy<=m.get('run',0),'Union occupancy')
        need(m.get('both_busy',0)<=min(m.get('lane0_busy',0),m.get('lane1_busy',0)),'Overlap occupancy')
        if lanes==1:need(m.get('lane1_busy')==0 and m.get('both_busy')==0,'Scalar second lane')
        if c['status']==0:
            blocks=c['rows']*c['columns']//32
            need(m.get('block_ops')==blocks,'Successful block count')
            need(m.get('r_beats',0)*64==blocks*34,'Successful payload bytes')
            # Busy-overwrite scenario intentionally presents activations that
            # cannot handshake; every valid activation is still accepted once.
            need(m.get('activation_writes')==c['columns']//32,'Activation count')
            if lanes==2 and c['rows']>=2:
                need(m.get('lane1_busy',0)>0 and m.get('both_busy',0)>0,'Second lane not exercised')
            for k in totals:totals[k]+=m.get(k,0)
            cycles+=c['cycles'];successful+=1
    any_busy=totals['lane0_busy']+totals['lane1_busy']-totals['both_busy']
    return dict(status='FAIL' if errors else 'PASS',errors=errors,lanes=lanes,metrics=metrics,
                successful_jobs=successful,successful_job_cycles=cycles,successful_totals=totals,
                any_lane_busy_fraction=any_busy/cycles if cycles else 0,
                q8_stall_fraction=totals['q8_stall']/cycles if cycles else 0,
                useful_bytes_per_cycle=totals['r_beats']*64/cycles if cycles else 0,
                scope='RTL simulation. Phase counters partition job cycles; busy/stall/event counters may overlap.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--lanes',type=int,choices=[1,2],required=True);p.add_argument('--json',type=Path)
    a=p.parse_args();r=review(a.log.read_text(errors='replace'),a.lanes)
    if a.json:
        with a.json.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r));return int(r['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
