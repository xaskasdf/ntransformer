"""Check modeled traces without rerunning the scheduler or trusting its PASS."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)


def peak(intervals):
    deltas = defaultdict(int)
    for start,end in intervals:
        need(end>start,'positive lifetime')
        deltas[start] += 1
        deltas[end] -= 1
    used = high = 0
    for cycle in sorted(deltas):
        used += deltas[cycle]
        high = max(high,used)
        need(used>=0,'resource balance')
    need(used==0,'resource drain')
    return high


def review_case(case):
    c,m,rows,jobs = (case[k] for k in ('config','metrics','trace','jobs'))
    n = case['count']; pause = case['pause_us']*200
    policy=case['policy']
    need(c==dict(channels=2,dedicated=2,reserve=c['reserve'],bank_bytes=4096,beat_bytes=64,
                 outstanding=4,descriptors=16,result_slots=16,b_delay=32,read_ports=2,
                 reserve_read_ports=2,reserve_quota=None,release_on_b=False,clock_mhz=200,
                 ingress='ordered' if policy=='ordered' else 'eligible',
                 ingress_window={'ordered':None,'window2':2,'window8':8,'unbounded':None}[policy]),
         'fixed model configuration')
    need(case['stalls']==dict(aw=[],w=[[0,0,pause]] if case['pause_kind']=='w' else [],
                             b=[[0,0,pause]] if case['pause_kind']=='b' else [],
                             result=[],w_period=0,w_closed=0),'pause configuration')
    need(jobs==[dict(tag=i,channel=i%2 if case['split']=='balanced' else int(i%4==3)
                    if case['split']=='3to1' else int(i>=n//2),beats=64,available=0)
                for i in range(n)],'exact offered workload')
    need(len(rows)==len(jobs)==n and [j['tag'] for j in jobs]==list(range(n)), 'all jobs')
    need(len({r['tag'] for r in rows})==n and {r['tag'] for r in rows}==set(range(n)), 'unique trace')
    remaining=list(range(n)); previous_fill=-1; banks=defaultdict(list)
    for row in rows:
        tag=row['tag']; job=jobs[tag]; ch=row['channel']; bank=row['bank']
        need(row['beats']==row['received_beats']==job['beats']==64 and ch==job['channel'],'payload conservation')
        need(0<=bank<4+c['reserve'] and (bank>=4 or bank//2==ch),'bank ownership')
        window=1 if c['ingress']=='ordered' else c['ingress_window']
        need(tag in (remaining if window is None else remaining[:window]),'descriptor visibility')
        need(tag==next(t for t in remaining if jobs[t]['channel']==ch),'destination FIFO')
        remaining.remove(tag)
        need(row['admitted']>=job['available'] and row['admitted']>previous_fill,'single nonpreemptive ingress')
        need(row['filled']==row['admitted']+63,'input beat count')
        previous_fill=row['filled']
        need(row['aw']>row['filled'] and row['first_w']>=row['aw'],'store before forward')
        need(row['last_w']==row['first_w']+63 and row['bank_freed']==row['last_w'],'output beat lifetime')
        need(row['b']>=row['last_w']+32 and row['complete']>row['b'],'response causality')
        if ch==0 and case['pause_kind']=='w':need(row['first_w']>=pause,'W pause enforced')
        if ch==0 and case['pause_kind']=='b':need(row['b']>=pause,'B pause enforced')
        banks[bank].append((row['admitted'],row['bank_freed']))
    for intervals in banks.values():
        need(peak(intervals)==1,'bank reused before release')
    for ch in (0,1):
        q=[r for r in rows if r['channel']==ch]
        need(all(a['last_w']<b['first_w'] and a['b']<b['b'] for a,b in zip(q,q[1:])),'ordered egress')
        need(peak([(r['admitted'],r['b']) for r in q])<=16,'descriptor credits')
        need(peak([(r['admitted'],r['complete']) for r in q])<=16,'result credits')
        need(peak([(r['aw'],r['b']) for r in q])<=4,'outstanding credits')
    need(m['drained'] and m['completed']==m['admitted']==m['offered']==n,'full drain')
    need(m['cycles']==max(r['complete'] for r in rows)+1,'final cycle')
    need(m['payload_bytes']==n*4096 and m['bytes_by_channel']==
         [sum(r['beats']*64 for r in rows if r['channel']==ch) for ch in (0,1)],'byte totals')
    need(abs(m['gbps_including_fill_and_drain']-n*4096/(m['cycles']*5))<1e-12,'goodput arithmetic')
    need(m['peak_banks']==peak([(r['admitted'],r['bank_freed']) for r in rows]),'bank peak')
    need(m['peak_reserve']==peak([(r['admitted'],r['bank_freed']) for r in rows if r['bank']>=4]),'reserve peak')
    # All descriptors are available at zero; every pre-fill gap is blocked.
    blocked=rows[0]['admitted']+sum(b['admitted']-a['filled']-1 for a,b in zip(rows,rows[1:]))
    need(m['counters'].get('ingress_blocked_cycles',0)==blocked,'blocked cycle arithmetic')
    d=case['derived']
    need(d['blocked_ns']==blocked*5 and d['completion_ns']==m['cycles']*5,'derived times')
    need(d['peer_completions_before_release']==sum(r['channel']==1 and r['complete']<pause for r in rows),'peer progress')
    return dict(frames=n,bytes=n*4096)


def negatives(case):
    out=[]
    for name in ('lost_beat','early_reuse','hidden_descriptor','early_response','false_goodput'):
        c=copy.deepcopy(case)
        if name=='lost_beat':c['trace'][0]['received_beats']-=1
        elif name=='early_reuse':c['trace'][0]['bank_freed']-=1
        elif name=='hidden_descriptor':c['trace'][0],c['trace'][1]=c['trace'][1],c['trace'][0]
        elif name=='early_response':c['trace'][0]['b']=c['trace'][0]['last_w']
        else:c['metrics']['gbps_including_fill_and_drain']*=2
        try:review_case(c)
        except ValueError as e:out.append(dict(mutation=name,rejection=str(e)))
        else:raise ValueError('negative control missed '+name)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--report',default='REVIEW.json',help='New report filename inside the run')
    args=ap.parse_args();p=args.run/'RESULTS.json';report=json.loads(p.read_text())
    need(Path(args.report).name==args.report,'report must be a filename')
    for name,h in report['inputs_sha256'].items():
        need(hashlib.sha256((args.run/'inputs'/name).read_bytes()).hexdigest()==h,'source snapshot')
    expected={(n,s,k,u,r,w) for n in (16,128) for s in ('balanced','3to1','clustered')
              for k,u in [('none',0)]+[(k,u) for k in ('w','b') for u in (4,8,16)]
              for r in (4,6,8) for w in ('ordered','window2','window8','unbounded')}
    got={(x['count'],x['split'],x['pause_kind'],x['pause_us'],x['config']['reserve'],x['policy']) for x in report['cases']}
    need(got==expected and report['case_count']==len(report['cases'])==len(expected),'exact sweep coverage')
    totals=[review_case(c) for c in report['cases']]
    result=dict(status='BOUNDED_LOOKAHEAD_MODEL_REVIEW_PASS_NOT_RTL',cases=len(totals),
                frames=sum(x['frames'] for x in totals),bytes=sum(x['bytes'] for x in totals),
                negative_controls=negatives(report['cases'][0]),
                results_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (args.run/args.report).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
