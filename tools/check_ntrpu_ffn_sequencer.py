"""Independent phase/count review of the FFN descriptor protocol bank (no MAC)."""
import re


def contracts(fp32=False):
    down_bytes=4 if fp32 else 8
    rows=[]
    def add(label,d=64,h=96,status=0,child=0,phase=0,stages=0,offers=0,cancels=0,stage_bytes=0,bytes=0):
        rows.append(dict(id=len(rows)+1,label=label,dim=d,hidden=h,status=status,child=child,
                         phase=phase,stages=stages,offers=offers,cancels=cancels,
                         stage_bytes=stage_bytes,bytes=bytes))
    def valid(label,d,h):
        add(label,d,h,phase=3,stages=4,offers=4,bytes=d*down_bytes,stage_bytes=16*h+((h+31)//32)*64+d*down_bytes)
    for label,d,h in [('small',32,32),('unequal',64,96),('brandon',256,720),
                      ('llama',2048,8192),('reverse',8192,2048),('maximum',8192,8192),('tail',33,1024)]:
        valid(label,d,h)
    for mode in range(13):add(f'shape_{mode}',status=2 if mode==11 else 1)
    for b in range(8):
        for mode in range(7):add(f'buffer_{b}_{mode}',status=2 if mode==4 else 1)
    for b in range(4,8):
        for other in range(8):
            if b!=other:add(f'alias_{b}_{other}',status=1)
    payloads=[768,768,192,64*down_bytes]
    for p in range(4):
        for f in range(1,11 if p==2 else 10):
            partial=(64 if p==2 else 8) if f==2 else 0
            add(f'fault_{p}_{f}',status=6 if f==1 else 8 if f==2 else 5,
                child=3 if f in {1,7} else 8 if f==2 else 0,phase=p,stages=p,offers=p+1,
                stage_bytes=sum(payloads[:p])+partial,bytes=partial if p==3 else 0)
    add('cancel_preflight',status=7,cancels=1)
    for p in range(4):
        for mode in range(3):add(f'cancel_{p}_{mode}',status=7,phase=p,stages=p,offers=p+1,cancels=1,
                               stage_bytes=sum(payloads[:p+1]),bytes=64*down_bytes if p==3 else 0)
    for p in range(3):add(f'cancel_between_{p}',status=7,phase=p,stages=p+1,offers=p+1,cancels=1,
                         stage_bytes=sum(payloads[:p+1]))
    for p in range(4):
        for running in range(2):rows.append(dict(id=len(rows)+1,phase=p,running=running,offers=p+running))
    valid('recovery',64,96)
    return rows


def review(text,seed,fp32=False):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    def records(prefix):
        return [{k:int(v) if v.isdigit() else v for k,v in re.findall(r'(\w+)=(\w+)',line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_FFN_SEQ_SEED')==[dict(seed=seed)],'Seed')
    if fp32:need(records('NTRPU_FFN_SEQ_FORMAT')==[dict(fp32=1)],'FP32 output format')
    cases=records('NTRPU_FFN_SEQ_CASE');resets=records('NTRPU_FFN_SEQ_RESET')
    expected=contracts(fp32);actual=sorted(cases+resets,key=lambda r:r.get('id',0))
    need([r.get('id') for r in actual]==list(range(1,167)),'Case identities')
    for want,got in zip(expected,actual):
        need(all(got.get(k)==v for k,v in want.items()),'Case contract '+str(want['id']))
        if 'label' in want:need(0<got.get('cycles',0)<20000,'Case latency')
    children=[dict(case=r['id'],phase=p,sequence=(0xfffffffe+p)&0xffffffff)
              for r in expected for p in range(r['offers'])]
    need(records('NTRPU_FFN_SEQ_CHILD')==children,'Child phase/sequence trace')
    summaries=records('NTRPU_FFN_SEQ_PASS');need(len(summaries)==1,'Summary count')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==166 and summary.get('resets')==8 and summary.get('cancels')==16 and
         summary.get('offers')==len(children)==177,'Coverage totals')
    for k in ['job_stalls','done_stalls']:need(summary.get(k,0)>0,'Backpressure '+k)
    need(summary.get('cycles',0)>sum(r.get('cycles',0) for r in cases),'Total cycles including resets/retirement')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,summary=summary,cases=actual,
                scope='Protocol stubs only; no AXI payload or numerical FFN validated.')
