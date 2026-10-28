"""Check observed bind/command precedence and drain, with no arithmetic claim."""
from collections import Counter
import re
SCHEMA={'RESET':'cycle id','BIND':'cycle id sequence epoch capacity',
 'BIND_ACK':'cycle id status sequence epoch','ADMIT':'cycle job mode sequence epoch position',
 'DONE':'cycle job status rope capture committed','ACK':'cycle job status sequence epoch',
 'RELEASE':'cycle job','PASS':'seed binds bind_acks jobs acks errors resets held blocked cycles'}
def need(ok,why):
    if not ok:raise ValueError(why)
def parse(text):
    need(not re.search(r'(?m)^(?:Fatal:|FATAL:|ERROR:|FAIL)',text),'Failure diagnostic')
    events=[]
    for line in text.splitlines():
        if not line.startswith('RA_'):continue
        tag,*fields=line[3:].split();need(tag in SCHEMA,'Known event')
        pairs=[x.split('=') for x in fields]
        need(all(len(x)==2 for x in pairs) and [x[0] for x in pairs]==SCHEMA[tag].split(),'Exact schema')
        row={}
        for key,val in pairs:
            need(re.fullmatch('[0-9a-f]{16}' if key in ('sequence','epoch') else '[0-9]+',val) is not None,'Numeric shape')
            row[key]=int(val,16 if key in ('sequence','epoch') else 10)
        events.append((tag,row))
    return events
def review_events(events,seed):
    counts=Counter(t for t,r in events)
    need(counts==dict(RESET=2,BIND=15,BIND_ACK=14,ADMIT=6,DONE=6,ACK=6,RELEASE=6,PASS=1),'Complete event inventory')
    need(events[0][0]=='RESET' and events[-2][0]=='RESET' and events[-1][0]=='PASS','Reset/summary boundaries')
    cycles=[r['cycle'] for t,r in events if t!='PASS'];need(all(a<b for a,b in zip(cycles,cycles[1:])),'Strict causal event order')
    by={t:[r for tag,r in events if tag==t] for t in SCHEMA};s=events[-1][1]
    need(s['seed']==seed and s['cycles']==cycles[-1],'Summary identity')
    need({k:s[k] for k in ('binds','bind_acks','jobs','acks','errors','resets')}==dict(binds=15,bind_acks=14,jobs=6,acks=6,errors=6,resets=2),'Summary counts')
    base_seq=0x1234567800000000;base_epoch=0xfedcba9800000001
    held=0;blocked=0
    for i,b in enumerate(by['BIND'],1):
        if i==1:seq=base_seq+0x99;epoch=base_epoch;capacity=0
        elif i==14:seq=base_seq+0xab;epoch=base_epoch+12;capacity=4
        elif i==15:seq=base_seq+0xaa;epoch=base_epoch+13;capacity=4
        else:
            job=i//2;seq=(base_seq+job)^(1<<40 if i%2 else 0);epoch=base_epoch+i-1;capacity=4
        need(b==dict(cycle=b['cycle'],id=i,sequence=seq,epoch=epoch,capacity=capacity),'Frozen bind identity')
        if i==15:continue
        a=by['BIND_ACK'][i-1];want=1 if i==1 else 2 if i==14 else 0
        need(a==dict(cycle=a['cycle'],id=i,status=want,sequence=seq,epoch=epoch),'Only owned bind completion acknowledged')
        latency=1 if want else 3
        need(a['cycle']>b['cycle']+latency,'Bind completion held before acknowledgment')
        held+=a['cycle']-b['cycle']-latency
    for i,a in enumerate(by['ADMIT'],1):
        mode=(i-1)%3;want=(2,1,11)[mode];seq=base_seq+i;epoch=base_epoch+2*i-1
        need(a==dict(cycle=a['cycle'],job=i,mode=mode,sequence=seq,epoch=epoch,position=131072 if mode==1 else 0),'Frozen admitted command')
        done=by['DONE'][i-1];ack=by['ACK'][i-1];release=by['RELEASE'][i-1]
        need(done==dict(cycle=done['cycle'],job=i,status=want,rope=want,capture=want,committed=0),'Rejection provenance and no commit')
        need(ack==dict(cycle=ack['cycle'],job=i,status=want,sequence=seq,epoch=epoch),'Public completion identity')
        need(release['job']==i,'Drain belongs to current command')
        prior=by['BIND_ACK'][2*i-1];following=by['BIND'][2*i]
        need(prior['cycle']<a['cycle']<done['cycle']<ack['cycle']<release['cycle']<following['cycle'],'Bind ACK, command, held completion, drain, then rebind')
        blocked+=release['cycle']-a['cycle']
    need(s['held']==held and s['blocked']==blocked,'Hold/exclusion cycles from observed lifetimes')
    need([r['id'] for r in by['RESET']]==[1,2] and by['BIND'][-1]['cycle']<by['RESET'][-1]['cycle'],'Coordinated reset drops final unacknowledged bind')
    return dict(status='PASS_ROPE_ATTENTION_ADMISSION_R1_TRACE',summary=s,arithmetic_executed=False,
        successful_inference_commands=0,scope='Actual admission and rejection hardware; arithmetic sentinels reject any start.')
def review(text,seed):return review_events(parse(text),seed)
def controls(text,seed):
    events=parse(text);review_events(events,seed);rejected=[]
    def reject(label,bad):
        try:review_events(bad,seed)
        except (ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted '+label)
    for tag in SCHEMA:
        i=next(i for i,(t,r) in enumerate(events) if t==tag)
        reject('missing-'+tag,events[:i]+events[i+1:]);reject('duplicate-'+tag,events[:i]+[events[i]]+events[i:])
        for key in events[i][1]:
            row=dict(events[i][1]);row[key]^=(1<<40 if key in ('sequence','epoch') else 1<<31 if key in ('cycle','cycles') else 1)
            bad=events.copy();bad[i]=(tag,row);reject(tag+'-'+key,bad)
    for job in range(1,7):
        release=next(r for t,r in events if t=='RELEASE' and r['job']==job)
        i=next(i for i,(t,r) in enumerate(events) if t=='BIND' and r['id']==2*job+1)
        row=dict(events[i][1]);row['cycle']=release['cycle']-1
        bad=events.copy();bad[i]=('BIND',row);reject('rebind-before-drain-'+str(job),bad)
    return rejected
