"""Independent contract review for the descriptor-only matrix control bank."""
import re

CASES=[(1,32,32,0,0),(31,64,32,0,0),(32,9,4,0,0),(33,64,32,0,0),
 (720,8,4,0,0),(8192,64,32,0,0),(2048,256,32,0,0),(8191,256,32,0,0),(8192,256,1,0,0),
 (64,64,32,1,3),(64,64,32,2,3),*[(64,64,32,m,5) for m in [4,5,6,7]],
 *[(64,64,32,m,7) for m in range(8,13)],(64,64,32,13,0),(32,32,32,14,0),(33,64,32,0,0),
 (0,32,32,0,1),(8193,32,32,0,1),(32,0,32,0,1),(32,257,32,0,1),(32,32,0,0,1),(32,32,33,0,1),(33,33,32,0,1),
 *[(64,64,32,m,2 if m in {27,28} else 1) for m in range(20,34)],(64,64,32,0,0)]


def review(text,seed):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    def records(prefix):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_MATRIX_SEED')==[dict(seed=seed)],'Seed')
    cases=records('NTRPU_MATRIX_CASE');groups=records('NTRPU_MATRIX_GROUP')
    need([c.get('id') for c in cases]==list(range(1,46)),'Case identities')
    expected_groups=[]
    for c,contract in zip(cases,CASES):
        n,k,t,mode,status=contract;ident=c['id']
        need(tuple(c.get(x) for x in ['rows','blocks','chunk','mode','status'])==contract,'Contract '+str(ident))
        need(0<c.get('cycles',0)<300000 and c.get('reset')==int(mode in {13,14}),'Latency/reset')
        actual=tuple(c.get(x) for x in ['groups','tiles','bytes','completed_rows'])
        if mode==0 and status==0:expected=((n+31)//32,((n+31)//32)*((k+t-1)//t),n*8,n)
        elif status in {1,2} or mode==8:expected=(0,0,0,0)
        elif mode==1:expected=(1,2,64,0)
        elif mode==2:expected=(2,4,320,32)
        elif mode in {4,5}:expected=(2,4,512,32)
        elif mode==6:expected=(2,4,504,32)
        elif mode==7:expected=(1,1,256,0)
        elif mode==9:
            expected=(1,c.get('tiles'),0,0);need(0<=c.get('tiles',-1)<=1,'Cancel waiting command')
        elif mode==10:
            expected=(1,c.get('tiles'),0,0);need(1<=c.get('tiles',-1)<=2,'Cancel active tile')
        elif mode in {11,12}:expected=(1,2,256,32)
        elif mode==13:expected=(1,1,0,0)
        elif mode==14:expected=(1,1,256,32)
        else:raise AssertionError(contract)
        need(actual==expected,'Engine counts '+str(ident))
        for g in range(expected[0]):
            expected_groups.append(dict(case=ident,index=g,first_row=g*32,rows=min(32,n-g*32),sequence=(0xfffffff0+ident+g)&0xffffffff))
    need(groups==expected_groups,'Group shape/address identity trace')
    summaries=records('NTRPU_MATRIX_PASS');need(len(summaries)==1,'Summary count')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==45 and summary.get('resets')==2 and summary.get('cancels')==5,'Coverage totals')
    for x in ['cycles','groups','tiles']:need(summary.get(x)==sum(c.get(x,0) for c in cases),'Total '+x)
    for x in ['job_stalls','tile_stalls','done_stalls']:need(summary.get(x,0)>0,'Backpressure '+x)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=cases,summary=summary,
                scope='Descriptor controller with protocol stub; no MAC values or AXI data validated by this bank')
