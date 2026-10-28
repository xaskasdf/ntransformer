"""Review actual lease RTL traces driven by synthetic producer/consumer peers."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re
from pathlib import Path
from ntrpu_qkv_rope_plan_r1 import plan

SCHEMA={
 'INPUT':'cycle job index address base bytes','BOUNDARY':'cycle job index address base bytes',
 'ADMIT':'cycle job mode sequence epoch layer position','REGION':'cycle job index address base bytes',
 'CHILD':'cycle job sequence epoch layer position lease','CANCEL':'cycle job','ACK':'cycle job',
 'CHILD_ACK':'cycle job','SOURCE_ACK':'cycle job','RESET':'cycle job id',
 'INJECT':'cycle job phase mode','REPLY':'cycle job status committed bytes tokens','RESTORE':'cycle job phase',
 'COMPLETE':'cycle job status source child committed bytes tokens','WORD':'cycle job index data',
 'AUTH':'cycle job revoked valid q_valid child_ce status','RELEASE':'cycle job reset',
 'PASS':'seed jobs admissions children child_acks source_acks public_acks resets words cycles'}
HEX={'address','base','sequence','epoch','lease','data'}
def need(ok,why):
    if not ok:raise ValueError(why)
def parse(text):
    need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|QRL_FAIL|FAIL[: ])',text),'Failure diagnostic')
    events=[]
    for line in text.splitlines():
        if not line.startswith('QRL_'):continue
        tag,*fields=line[4:].split();need(tag in SCHEMA,'Known trace event')
        need([x.split('=')[0] for x in fields]==SCHEMA[tag].split(),'Event schema');row={}
        for field in fields:
            key,value=field.split('=');hexa=key in HEX or (key=='bytes' and tag in ('INPUT','BOUNDARY','REGION'))
            need(re.fullmatch('[0-9a-f]+' if hexa else '[0-9]+',value),'Known finite field');row[key]=int(value,16 if hexa else 10)
        events.append((tag,row))
    return events
def regions(mode):
    spans=[8192,8192,4096,4456448,1114112,1114112,8192,2048,2048,256]
    rows=[dict(index=i,address=((i+1)<<40)+4032,base=(i+1)<<40,bytes=((n+4032+4095)//4096)*4096) for i,n in enumerate(spans)]
    if 100<=mode<170:
        i,kind=divmod(mode-100,7);r=rows[i]
        if kind==0:r['address']+=1
        elif kind==1:r['base']+=1
        elif kind==2:r['bytes']-=1
        elif kind==3:r['bytes']=0
        elif kind==4:r['bytes']=64
        elif kind==5:r['address']=r['base']-64
        else:r.update(base=0xfffffffffffff000,address=0xffffffffffffffc0,bytes=8192)
    if 200<=mode<210:rows[mode-200].update(base=0xe00000000000,address=0xe00000000fc0)
    if 300<=mode<340:
        w,other=divmod(mode-300,10);slot=[2,6,7,8][w]
        if slot!=other:
            rows[slot].update(base=rows[other]['base'],address=rows[other]['base']+4032,bytes=8388608);rows[other]['bytes']=8388608
    if mode==20:rows[1]=dict(rows[0],index=1)
    return rows

def review_events(events,seed=1):
    need(events and events[0][0]=='RESET' and events[-1][0]=='PASS','Complete suite boundaries')
    counts=Counter(t for t,r in events);summary=events[-1][1];specs=plan()
    need(counts['PASS']==1 and summary['seed']==seed and summary['jobs']==len(specs)==348,'Exact planned suite')
    aliases=dict(admissions='ADMIT',children='CHILD',child_acks='CHILD_ACK',source_acks='SOURCE_ACK',public_acks='ACK',resets='RESET',words='WORD')
    need(all(summary[k]==counts[t] for k,t in aliases.items()),'Summary versus observations')
    need(summary['admissions']==348 and summary['public_acks']==summary['source_acks']==346 and summary['resets']==3,'Whole command coverage')
    need(summary['children']==sum(s['child'] for s in specs) and summary['child_acks']==summary['children']-2,'Child retirement coverage')
    need([r['id'] for t,r in events if t=='RESET']==[1,2,3],'Coordinated reset sequence')
    previous=-1;groups=defaultdict(list)
    for tag,row in events[:-1]:
        need(row['cycle']>=previous,'Monotonic cycle');previous=row['cycle']
        if row['job']==0:need(tag=='RESET' and row['id']==1,'Initial reset only')
        else:need(1<=row['job']<=348,'Job range');groups[row['job']].append((tag,row))
    need(previous==summary['cycles']<1000000,'Terminal cycle and global deadline');metrics=[];prior_release=events[0][1]['cycle']
    for job,spec in enumerate(specs,1):
        ev=groups[job];by=defaultdict(list)
        for t,r in ev:by[t].append(r)
        def one(tag,count=1):
            need(len(by[tag])==count,'Exact '+tag+' extent');return by[tag][0] if count else None
        admit=one('ADMIT');release=one('RELEASE');mode=spec['mode'];seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        layer=16 if mode==10 else 0x10000000 if mode==52 else job%16;pos=131072 if mode==53 else job%4
        need(admit==dict(cycle=admit['cycle'],job=job,mode=mode,sequence=seq,epoch=epoch,layer=layer,position=pos),'Planned admission identity')
        need(ev[0][0]=='INPUT' and ev[-1][0]=='RELEASE' and ev[0][1]['cycle']>=prior_release and release['cycle']>admit['cycle'],'Serialized ownership')
        prior_release=release['cycle'];need(release['reset']==int(spec['reset']),'Planned reset boundary')
        expected=regions(mode)
        for tag in ('INPUT','BOUNDARY'):
            need(len(by[tag])==10,'Ten retained region observations')
            for i,row in enumerate(by[tag]):
                need({k:row[k] for k in ('index','address','base','bytes')}==expected[i],'Actual descriptor boundary matches intended stimulus')
                need(row['cycle']<admit['cycle'] if tag=='INPUT' else row['cycle']==admit['cycle'],'Input capture clock')
        child=one('CHILD',int(spec['child']));one('REGION',4 if child else 0)
        if child:
            need(child==dict(cycle=child['cycle'],job=job,sequence=seq,epoch=epoch,layer=layer,position=pos,lease=0x2000000010000000+job*64),'Child descriptor identity')
            need(child['cycle']>admit['cycle'],'Child after retained admission')
            for i,row in enumerate(by['REGION']):
                need(row['cycle']==child['cycle'] and {k:row[k] for k in ('index','address','base','bytes')}==dict(expected[i+6],index=i),'Published physical regions derive from captured allocation')
        reply=one('REPLY',int(spec['child'] and mode!=39))
        if reply:
            failed=mode==1 or (spec['status']!=0 and mode not in (2,29,46))
            need(reply==dict(cycle=reply['cycle'],job=job,status=(3 if mode==1 else 7) if failed else 0,
                committed=int(not failed and mode!=46),bytes=128 if mode==2 else 4096,tokens=pos if failed else pos+1),'Synthetic reply stimulus')
            need(reply['cycle']>child['cycle'],'Reply after command')
        complete=one('COMPLETE',int(mode!=39))
        if complete:
            need(complete==dict(cycle=complete['cycle'],job=job,status=spec['status'],source=6 if mode==3 else 0,
                child=reply['status'] if reply else 0,committed=reply['committed'] if reply else 0,
                bytes=reply['bytes'] if reply else 0,tokens=reply['tokens'] if reply else 0),'Preserve status and observed append certificate')
            need(complete['cycle']>(reply or admit)['cycle'],'Completion after owned reply/rejection')
        ack=one('ACK',int(not spec['reset']));ca=one('CHILD_ACK',int(spec['child'] and not spec['reset']));sa=one('SOURCE_ACK',int(not spec['reset']))
        if ack:
            need(ack['cycle']>=complete['cycle']+7 and sa['cycle']>ack['cycle'],'Public ACK before producer release')
            if ca:need(ack['cycle']<ca['cycle'] and sa['cycle']>=ca['cycle']+(40 if mode==37 else 5+seed%4),'Full consumer drain precedes producer ACK')
            need(release['cycle']>=sa['cycle']+(40 if mode==38 else 7+seed%4),'Producer drain precedes next command')
        if spec['reset']:
            reset=one('RESET');need(reset['cycle']>(complete or child)['cycle'] and release['cycle']>=reset['cycle'],'Coordinated reset retires without ACK')
        else:one('RESET',0)
        count=2048 if job==1 else 8 if spec['status']==0 and not spec['reset'] else 0
        need(len(by['WORD'])==count,'Exact public read coverage')
        for i,row in enumerate(by['WORD']):
            index=(i*257+31)%2048;word=(0x3f000000^(seq&0xffffffff)^(index*0x00010203))&0xffffffff
            need(row['index']==index and row['data']==word and complete['cycle']<row['cycle']<ack['cycle'],'Owned synchronous result mapping')
        auth=one('AUTH',int(spec['late']))
        if auth:need(auth==dict(cycle=auth['cycle'],job=job,revoked=1,valid=0,q_valid=0,child_ce=0,status=0) and complete['cycle']<auth['cycle']<ack['cycle'],'Sticky revocation hides RAM/results')
        phases=([0] if mode in (21,26,27,45) else [])+([1] if child and mode!=39 else [])+([2] if spec['late'] else [])
        need([r['phase'] for r in by['INJECT']]==phases and all(r['mode']==mode for r in by['INJECT']),'Planned injections')
        for row in by['INJECT']:
            if row['phase']==0:need(admit['cycle']<row['cycle']<complete['cycle'],'Offer fault timing')
            if row['phase']==1:need(child['cycle']<row['cycle']<reply['cycle'],'Owned consumer fault timing')
            if row['phase']==2:need(by['WORD'][-1]['cycle']<row['cycle']<auth['cycle'],'Late fault after visible results')
        restores=([1] if mode==29 else [])+([2] if spec['late'] else [])+([3] if mode==35 else [])+([4] if mode in (7,8,23,36) else [])
        need([r['phase'] for r in by['RESTORE']]==restores,'Planned ownership recovery')
        for row in by['RESTORE']:
            if row['phase']==1:need(reply['cycle']+9<=row['cycle']<complete['cycle'],'Foreign child held before completion')
            if row['phase']==2:need(auth['cycle']==row['cycle']<ack['cycle'],'Restored cause remains revoked')
            if row['phase']==3:need(ack['cycle']+9<=row['cycle']<ca['cycle'],'Foreign child cannot be ACKed')
            if row['phase']==4:need(ack['cycle']+18<=row['cycle']<sa['cycle'],'Foreign producer cannot be ACKed')
        cancels=by['CANCEL'];expected_cancel=mode in (22,23,24,25,28,34,41,42,43,44)
        need(len(cancels)==int(expected_cancel),'One cancellation per owned consumer fault')
        if cancels:need(child['cycle']<cancels[0]['cycle']<(ack or release)['cycle'],'Cancellation ownership')
        metrics.append(dict(job=job,mode=mode,status=spec['status'],child=int(spec['child']),reset=spec['reset'],words=count))
    return dict(status='PASS_QKV_ROPE_HANDOFF_R1_NATIVE_PROTOCOL',summary=summary,metrics=metrics,
        scope='Actual lease controller with synthetic producer and consumer replies; no arithmetic or memory traffic executed.')
def review(text,seed=1):return review_events(parse(text),seed)
def controls(text,seed=1):
    events=parse(text);review_events(events,seed);rejected=[]
    for tag in SCHEMA:
        target=next(i for i,(t,r) in enumerate(events) if t==tag)
        for field in SCHEMA[tag].split():
            if field=='cycle':continue
            mutation=events[:];row=dict(events[target][1]);row[field]^=1;mutation[target]=(tag,row)
            try:review_events(mutation,seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.'+field)
            else:raise AssertionError('Accepted corruption '+tag+'.'+field)
        for kind in ('missing','duplicate'):
            mutation=events[:target]+([] if kind=='missing' else [events[target]]*2)+events[target+1:]
            try:review_events(mutation,seed)
            except (ValueError,AssertionError,KeyError,IndexError):rejected.append(tag+'.'+kind)
            else:raise AssertionError('Accepted '+kind+' '+tag)
    return rejected
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',action='store_true');a=p.parse_args()
    raw=a.log.read_bytes();text=raw.decode();result=review(text,a.seed)
    if a.controls:result['rejected_controls']=controls(text,a.seed)
    result.update(log_sha256=hashlib.sha256(raw).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],summary=result['summary'],controls=len(result.get('rejected_controls',[])))))
