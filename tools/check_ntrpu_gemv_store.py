"""Audit computed values separately from AXI-written private destination bytes."""
import math
import re
import struct
from check_ntrpu_gemv_memory import CASES as MEMORY_CASES, bursts

CASES=(MEMORY_CASES+[(17,32,m,3 if m<=42 else 8 if m<=45 else 0) for m in range(40,47)]+
       [(2,32,m,2 if m==49 else 7 if m>=52 else 1) for m in range(47,54)]+
       [(17,32,54,0),(17,32,55,0),(2,32,0,0)])


def review(text,seed,fixture=None):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(re.findall(r'^NTRPU_STORE_SEED seed=(\d+)\s*$',text,re.M)==[str(seed)],'Seed')
    def records(prefix):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',l)} for l in text.splitlines() if l.startswith(prefix+' ')]
    rows=records('NTRPU_STORE_CASE');writes=records('NTRPU_STORE_WRITE')
    contracts=[(j['rows'],j['k_blocks'],0,0) for j in fixture['jobs']] if fixture else CASES
    ids=list(range(1,len(contracts)+1))
    need([r.get('id') for r in rows]==ids and [r.get('id') for r in writes]==ids,'Case/write identities')
    expected_computed={};expected_written={};published=0
    for r,w,contract in zip(rows,writes,contracts):
        n,k,mode,status=contract;ident=r.get('id');reset=mode in {15,16,54,55}
        computed=(status==0 and not reset) or 40<=mode<=46 or mode in {54,55}
        need(tuple(r.get(f) for f in ['rows','blocks','scenario','status'])==contract,'Case contract '+str(ident))
        need(0<r.get('cycles',0)<(250000 if fixture else 20000),'Latency bound')
        need(r.get('outputs')==(n if computed else 0),'Private computed count')
        need((0<r.get('first_result',0)<r.get('cycles',0)) if computed else r.get('first_result')==0,'Compute timing')
        need(w.get('reset')==int(reset),'Reset record')
        if computed:
            if fixture:
                job=fixture['jobs'][ident-1];tiles=len(job['tiles'])
                wb=sum(bursts(t['address'],t['byte_count']) for t in job['tiles'])
            else:tiles=k//32;wb=tiles*bursts(4032,n*32*34)
            need((r.get('activation_beats'),r.get('activation_bursts'),r.get('weight_beats'),r.get('weight_bursts'),r.get('tiles'))==
                 (k,bursts(4032,k*64),n*k*34//64,wb,tiles),'Input traffic')
            written=8 if mode==54 else n if status==0 or mode==41 else 8
            aw=bursts(0 if reset else 4032,((written+7)//8)*64)
            need((w.get('aw'),w.get('w'),w.get('b'),w.get('bytes'))==(aw,(written+7)//8,0 if reset else aw,written*8),'Write traffic')
            if status==0 and not reset:published+=n
            for row in range(n):
                value=struct.unpack('>d',bytes.fromhex(job['reference_fp64_bits'][row]))[0] if fixture else float(sum(
                    ((b+lane)%5-2)*((b+row*3+lane)%7-3) for b in range(k) for lane in range(32)))
                expected_computed[(ident,row)]=value
                if row<written:expected_written[(ident,row)]=struct.pack('>d',value).hex()
        else:
            need(all(w.get(f)==0 for f in ['aw','w','b','bytes']),'Failed compute/descriptor wrote output')
            if mode in {1,2,3,4,9,47,48,49,50,51,52,53}:
                need(all(r.get(f)==0 for f in ['activation_beats','activation_bursts','weight_beats','weight_bursts','tiles']),'Rejected job read memory')
            elif 5<=mode<=12:
                ab,beats={5:(2,32),6:(2,32),7:(2,32),8:(1,1),10:(1,1),11:(2,65),12:(1,1)}[mode]
                need((r.get('activation_bursts'),r.get('activation_beats'))==(ab,beats),'Activation error drain')
                need(r.get('weight_beats')==r.get('weight_bursts')==r.get('tiles')==0,'Weights after bad activation')
            else:
                need(r.get('activation_beats')==(4 if mode==15 else k),'Reset/compute activation count')
                need(r.get('weight_beats')==0 if mode==15 else 0<r.get('weight_beats',0)<=n*k*34//64,'Reset/compute drain')
    computed_keys=[];written_keys=[]
    for line in text.splitlines():
        if line.startswith('NTRPU_STORE_COMPUTED '):
            f=dict(re.findall(r'(\w+)=([^ ]+)',line));key=(int(f['case']),int(f['row']))
            actual=float(f['value']);ref=float(f['reference']);value=expected_computed.get(key)
            need(value is not None and math.isfinite(actual) and actual==value and ref==value,'Computed oracle '+str(key));computed_keys.append(key)
        elif line.startswith('NTRPU_STORE_VALUE '):
            f=dict(re.findall(r'(\w+)=([^ ]+)',line));key=(int(f['case']),int(f['row']))
            need(f.get('bits')==expected_written.get(key),'Physical written oracle '+str(key));written_keys.append(key)
    need(computed_keys==list(expected_computed),'Computed identities/order')
    need(written_keys==list(expected_written),'Written identities/order')
    markers=re.findall(r'^NTRPU_STORE_PASS (.+)$',text,re.M);need(len(markers)==1,'Suite marker')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',markers[0])} if markers else {}
    need(summary.get('cases')==len(contracts) and summary.get('resets')==(0 if fixture else 4),'Suite cases/reset')
    resets=records('NTRPU_STORE_RESET')
    need(resets==([] if fixture else [dict(id=35,scenario=54,aw=1,w=1,b=0,bytes=64),
                                    dict(id=36,scenario=55,aw=1,w=3,b=0,bytes=136)]),'Write-reset phase evidence')
    for key in ['cycles','outputs','activation_bursts','weight_bursts']:need(summary.get(key)==sum(r.get(key,0) for r in rows),'Totals '+key)
    need(summary.get('ar_stalls',0)>0 and summary.get('done_stalls',0)>0,'Read/completion backpressure')
    need(sum(w.get('aw_stalls',0) for w in writes)>0 and sum(w.get('w_stalls',0) for w in writes)>0,'Write backpressure')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=rows,writes=writes,summary=summary,
                computed_values=len(computed_keys),written_private_values=len(written_keys),published_values=published)
