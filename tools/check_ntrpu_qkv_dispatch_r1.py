"""Independently reconcile snapshot bytes, KV writes and root/public ownership."""
import re

PLAN=[0,3,1,2]+list(range(10,28))+list(range(30,38))+list(range(40,49))+list(range(50,55))+list(range(60,64))
SCHEMA={
    'BEGIN':'job mode sequence epoch layer position base capacity expected',
    'ADMIT':'cycle job sequence epoch layer position lease',
    'APPEND':'cycle job','RAM':'cycle job address data','KV':'cycle job index data',
    'COMMIT':'cycle job status frames','AW':'cycle job address beats','W':'cycle job index data',
    'B':'cycle job beats response','KV_ACK':'cycle job status bytes tokens','ROOT':'cycle job prefix',
    'Q':'cycle job index data','HEAD':'cycle job head query','HEAD_ACK':'cycle job head',
    'INJECT':'cycle job kind kv q heads','COMPLETE':'cycle job status append root kv q bytes tokens committed',
    'RESULT':'cycle job index data','REVOKED':'cycle job mode','ACK':'cycle job',
    'SOURCE_ACK':'cycle job','RESET':'cycle job kv q heads aw w b',
    'RELEASE':'cycle job aw w b source_acks reset',
    'FAULT_PASS':'seed jobs frames heads words cycles errors revoked resets'}

def need(ok,why):
    if not ok:raise ValueError(why)

def status(mode):
    if 10<=mode<=14:return 1
    if 15<=mode<=19 or mode==46:return 2
    if mode in (20,22,23,24,25,47,48):return 11
    if mode in (21,27,36):return 5
    if mode==26 or 40<=mode<=45:return 7
    if mode in (30,31,37):return 4
    if mode in (32,33):return 8
    if mode in (34,35):return 3
    return 0

def word(index,job):
    mantissa=(index*131+job*977)&0x7fffff
    return [0,0x80000000,mantissa|1,0x007fffff,0x3f000000|mantissa,
        0xbf000000|mantissa,0x7f7fffff,0xff7fffff][index%8]

def frame(index,job,mode):
    data=sum(word(index*16+lane,job)<<(32*lane) for lane in range(16))
    if (mode==30 and index==128) or (mode==31 and index==127):data=(data&~0xffffffff)|0x7fc00001
    return data

def parse(text):
    need(not re.search(r'(?m)^(?:FATAL:|Fatal:|ERROR:|FAIL\b)',text),'Failure diagnostic')
    jobs={};summary=None;cycle=0
    for line in text.splitlines():
        if not line.startswith('DISPATCH_'):continue
        kind,*tokens=line[9:].split();need(kind in SCHEMA,'Known event')
        fields=[t.split('=') for t in tokens]
        need(all(len(f)==2 for f in fields),'Field syntax')
        need([f[0] for f in fields]==SCHEMA[kind].split(),'Exact event fields')
        row={}
        for key,value in fields:
            if key=='kind':need(value in ('cancel','epoch','source_valid','source_identity'),'Fault kind');row[key]=value;continue
            width=512 if key=='query' else 128 if key=='data' and kind!='RESULT' else 8 if key=='data' else 16 if key in ('sequence','epoch','base','lease') or (key=='address' and kind=='AW') else 0
            need(re.fullmatch('[0-9a-f]{'+str(width)+'}' if width else '[0-9]+',value) is not None,'Canonical numeric field')
            row[key]=int(value,16 if width else 10)
        if 'cycle' in row:need(row['cycle']>=cycle,'Monotonic event clock');cycle=row['cycle']
        need(summary is None,'No events after summary')
        if kind=='FAULT_PASS':summary=row;continue
        if kind=='BEGIN':
            need(row['job']==len(jobs)+1,'Ordered job admission');jobs[row['job']]=[]
        need(row['job']==len(jobs) and row['job'] in jobs,'Current job')
        jobs[row['job']].append((kind,row))
    need(len(jobs)==len(PLAN)==48 and summary is not None,'Complete scenario inventory')
    return jobs,summary

def review_job(job,events):
    mode=PLAN[job-1];want=status(mode)
    seq=0x1234567800000000+(job-1 if mode==3 else job)
    epoch=0xfedcba9800000000+(job-1 if mode==3 else job)
    capacity=3 if mode==1 else 2
    layer=16 if mode==12 else 15 if mode==1 else 0
    position=capacity if mode==13 else 1 if mode in (3,14) else 0
    lease=0 if mode==10 else 0xffffffffffffffe1 if mode==11 else 0xffffffffffffffe0 if mode==2 else 0x2000000000001000+job*64
    base=0x900000000fc0
    location=lambda beat:base+((layer*16+beat//4)*capacity+position)*256+(beat%4)*64
    need(events and events[0]==('BEGIN',dict(job=job,mode=mode,sequence=seq,epoch=epoch,layer=layer,
        position=position,base=base,capacity=capacity,expected=want)),'Independent descriptor/status')
    by={k:[] for k in SCHEMA};kv=[];qs=[];reads=[];aws=[];ws=[];bs=[];heads=[];hacks=[];results=[]
    admitted=append=commit=kvack=root=complete=ack=release=reset=revoke=None
    pending=None;burst_words=[];planned=0;last_cycle=0;committed={}
    for kind,row in events[1:]:
        need(kind in by and row['job']==job and 'cycle' in row,'Event identity')
        need(row['cycle']>=last_cycle and release is None,'Ordered events before release');last_cycle=row['cycle']
        need(reset is None or kind=='RELEASE','No work after coordinated reset')
        by[kind].append(row)
        if kind=='ADMIT':
            need(admitted is None,'One command')
            need(row==dict(cycle=row['cycle'],job=job,sequence=seq,epoch=epoch,layer=layer,position=position,lease=lease),'Admitted full identity')
            admitted=row;continue
        need(admitted is not None and row['cycle']>admitted['cycle'],'Work follows admission')
        if kind=='APPEND':
            need(append is None and not reads and not kv and root is None and complete is None,'Single append before reading snapshot');append=row
        elif kind=='RAM':
            need(append is not None and complete is None and ack is None,'Private snapshot remains owned')
            index=len(qs) if root else 128+len(kv)
            need(index<192 and row['address']==index and row['data']==frame(index,job,mode),'Exact ordered source RAM frame')
            need(not reads or row['address']!=reads[-1]['address'],'Single RAM request per frame');reads.append(row)
        elif kind in ('KV','Q'):
            collection=kv if kind=='KV' else qs;offset=128 if kind=='KV' else 0
            need(complete is None and ack is None and (root is not None)==(kind=='Q'),'KV before root and Q after root')
            need(row['index']==len(collection)<(64 if kind=='KV' else 128),'Ordered accepted frames')
            need(reads and reads[-1]['address']==offset+len(collection) and row['cycle']>reads[-1]['cycle'],'Data follows source RAM request')
            need(row['data']==reads[-1]['data']==frame(offset+len(collection),job,mode),'Exact forwarded frame')
            need(all(((row['data']>>(lane*32+23))&255)!=255 for lane in range(16)),'Nonfinite frame never transferred')
            collection.append(row)
        elif kind=='COMMIT':
            need(append is not None and commit is None and root is None and not aws and not ws,'Single producer commit before physical write')
            need(row['frames']==len(kv) and (row['status']!=0 or len(kv)==64),'Explicit producer completeness')
            need(row['status'] in (0,want),'Commit conveys preexisting producer fault');commit=row
        elif kind=='AW':
            need(commit is not None and commit['status']==0 and root is None and complete is None and pending is None,'Write owns successful producer commit')
            address=location(planned);beats=min(4-planned%4,(4096-address%4096)//64)
            need(row['address']==address and row['beats']==beats and 0<beats<=4 and planned+beats<=64,'Physical layout and page boundary')
            pending=dict(start=planned,beats=beats,cycle=row['cycle']);planned+=beats;aws.append(row)
        elif kind=='W':
            need(commit is not None and commit['status']==0 and root is None and complete is None,'Write data after producer commit')
            need(row['index']==len(ws)<64 and len(burst_words)<4,'Single ordered write channel')
            need(row['data']==frame(128+len(ws),job,mode)==kv[len(ws)]['data'],'Physical write equals captured KV')
            burst_words.append(row);ws.append(row)
        elif kind=='B':
            need(pending is not None and row['cycle']>pending['cycle'] and len(burst_words)==pending['beats'],'Response owns complete AW/W burst')
            need(row['beats']==pending['beats'],'B burst accounting')
            need([r['index'] for r in burst_words]==list(range(pending['start'],pending['start']+pending['beats'])),'W belongs to this address burst')
            response=2 if (mode==34 and not bs) or (mode==35 and planned==64) else 0
            need(row['response']==response,'Expected B response stimulus')
            if response==0:committed.update({r['index']:r['data'] for r in burst_words})
            pending=None;burst_words=[];bs.append(row)
        elif kind=='KV_ACK':
            need(append is not None and kvack is None and root is None and complete is None and pending is None and not burst_words and len(aws)==len(bs),'Unique drained append reply')
            need(row['bytes']==sum(r['beats']*64 for r in bs if r['response']==0),'Only successful B bytes confirmed')
            need(row['status'] in (0,2,3,4,7,8,11),'Known child status')
            if row['status']==0:need(len(kv)==len(ws)==64 and row['bytes']==4096 and row['tokens']==position+1,'Complete successful append')
            else:need(row['tokens']==position,'Failed append preserves prefix')
            kvack=row
        elif kind=='ROOT':
            need(root is None and complete is None and kvack is not None and kvack['status']==0 and row['cycle']>kvack['cycle'],'Attention after accepted append ACK')
            need(row['prefix']==position+1 and len(committed)==64 and all(committed[i]==frame(128+i,job,mode) for i in range(64)),'Whole KV committed before root');root=row
        elif kind=='HEAD':
            need(root is not None and len(qs)==128 and len(hacks)==len(heads) and complete is None,'Ordered nonoverlapping heads after Q')
            need(row['head']==len(heads)<32,'Head extent')
            data=sum(word(len(heads)*64+i,job)<<(i*32) for i in range(64))
            need(row['query']==data,'Exact assembled head query');heads.append(row)
        elif kind=='HEAD_ACK':
            need(len(hacks)<len(heads)==len(hacks)+1 and row['head']==len(hacks) and row['cycle']>heads[-1]['cycle'],'Drain each accepted head');hacks.append(row)
        elif kind=='INJECT':
            need(len(by[kind])==1 and complete is None and 40<=mode<=48,'One deliberate active fault')
            expected_kind='cancel' if mode<=45 else 'epoch' if mode==46 else 'source_valid' if mode==47 else 'source_identity'
            need((row['kv'],row['q'],row['heads'],row['kind'])==(len(kv),len(qs),len(heads),expected_kind),'Fault placement accounting')
            if mode==40:need(append is None and not reads,'Cancel before append')
            if mode in (41,46):need(len(kv)==17 and not ws,'Fault in KV capture')
            if mode==42:need(pending is not None,'Cancel with accepted AW outstanding')
            if mode==43:need(kvack is not None and kvack['status']==0 and root is None,'Cancel after successful append ACK')
            if mode in (44,47,48):need(len(qs)==17,'Fault during Q')
            if mode==45:need(len(heads)==4 and len(hacks)==3,'Cancel with owned arithmetic child')
        elif kind=='COMPLETE':
            need(complete is None and reset is None and pending is None and not burst_words and len(hacks)==len(heads),'Unique completion after child drain')
            need(row['status']==want and row['kv']==len(kv) and row['q']==len(qs),'Exact status and accepted extents')
            need((row['append'],row['bytes'],row['tokens'])==((kvack['status'],kvack['bytes'],kvack['tokens']) if kvack else (0,0,0)),'Retained child reply')
            committed_flag=int(kvack is not None and kvack['status']==0 and kvack['bytes']==4096 and kvack['tokens']==position+1)
            need(row['committed']==committed_flag,'KV commit survives downstream fault')
            if root is None:need(row['root']==0,'No invented root status')
            elif want==0:need(row['root']==0,'Successful root status')
            elif mode in (31,33,44,47,48):need(row['root']==7,'Dispatcher cancels root after upstream fault')
            elif mode in (36,37,45):need(row['root']==want,'Actual head failure/cancellation')
            if 10<=mode<=27:need(append is None and not reads and not ws and root is None,'Rejected admission has no side effects')
            if want==0:need(len(kv)==64 and len(qs)==128 and len(heads)==32 and len(reads)==192 and committed_flag,'Complete successful transaction')
            if mode==30:need(len(reads)==1 and not kv,'First nonfinite KV rejected')
            if mode==31:need(len(qs)==127 and len(reads)==192,'Last nonfinite Q rejected')
            if mode==32:need(len(reads)==1 and not kv,'KV RAM timeout')
            if mode==33:need(len(reads)==65 and not qs,'Q RAM timeout')
            if mode==34:need(len(bs)==1 and not committed,'First B failed')
            if mode==35:need(len(ws)==64 and bs[-1]['response']==2 and row['bytes']<4096,'Final B failed')
            if mode in (36,37):need(len(heads)==1,'Stop after first failed head')
            complete=row
        elif kind=='RESULT':
            need(complete is not None and want==0 and ack is None and revoke is None and len(results)<2048,'Authorized public result only')
            index=(len(results)*17+31)%2048
            need(row['index']==index and row['data']==(word(index,job)^0x100),'Exact permuted result bytes');results.append(row)
        elif kind=='REVOKED':
            need(50<=mode<=54 and row['mode']==mode and complete is not None and len(results)==2048 and revoke is None and ack is None,'Held-result revocation');revoke=row
        elif kind=='ACK':
            need(complete is not None and ack is None and row['cycle']>=complete['cycle']+6,'Public completion retained')
            need(len(results)==(2048 if want==0 else 0),'Readback before public ACK');ack=row
        elif kind=='SOURCE_ACK':
            need(mode not in (16,17,18,19,20) and ack is not None and len(by[kind])==1 and row['cycle']>ack['cycle'],'No early or foreign-source ACK')
            if mode in (48,54):need(row['cycle']>=ack['cycle']+6,'Foreign source held through public drain')
        elif kind=='RESET':
            need(60<=mode<=63 and reset is None and complete is None and ack is None and not by['SOURCE_ACK'],'Coordinated cancellation by reset')
            need((row['kv'],row['q'],row['heads'],row['aw'],row['w'],row['b'])==(len(kv),len(qs),len(heads),len(aws),len(ws),len(bs)),'Reset progress accounting')
            if mode==60:need(len(kv)==17 and not ws,'Reset during capture')
            if mode==61:need(pending is not None,'Reset with AW outstanding')
            if mode==62:need(len(qs)==17,'Reset during Q')
            if mode==63:need(len(heads)==len(hacks)==32,'Reset after all heads')
            reset=row
        elif kind=='RELEASE':
            need(row['aw']==len(aws) and row['w']==len(ws) and row['b']==len(bs) and row['source_acks']==len(by['SOURCE_ACK']),'Release accounting')
            need(row['reset']==int(reset is not None),'Release path classification')
            if reset:need(row['cycle']>reset['cycle'] and ack is None and not by['SOURCE_ACK'],'Coordinated reset release')
            else:
                need(ack is not None and pending is None and not burst_words and row['cycle']>ack['cycle'],'Completed transaction drains before reuse')
                owned=int(mode not in (16,17,18,19,20));need(len(by['SOURCE_ACK'])==owned,'Source ownership released exactly once')
                if owned:need(row['cycle']>by['SOURCE_ACK'][0]['cycle'],'Release after source ACK')
            release=row
        else:raise ValueError('Unexpected event '+kind)
    need(admitted is not None and release is not None,'Complete transaction lifetime')
    need((complete is not None)==(mode<60) and (reset is not None)==(mode>=60),'Scenario terminal')
    need(len(by['INJECT'])==int(40<=mode<=48) and len(by['REVOKED'])==int(50<=mode<=54),'Fault/revocation coverage')
    need(len(results)==(2048 if mode<60 and want==0 else 0),'Exact public result extent')
    return dict(job=job,mode=mode,frames=len(kv)+len(qs),heads=len(heads),words=len(results),
        resets=int(reset is not None),revoked=int(revoke is not None),errors=int(want!=0),
        aw=len(aws),w=len(ws),b=len(bs),committed_bytes=sum(r['beats']*64 for r in bs if r['response']==0),
        admission_cycle=admitted['cycle'],end_cycle=last_cycle)

def review_parsed(jobs,summary,seed):
    need(list(jobs)==list(range(1,49)),'Exact scenario identities')
    metrics=[review_job(j,e) for j,e in jobs.items()]
    for key in ('frames','heads','words','errors','revoked','resets'):
        need(summary[key]==sum(m[key] for m in metrics),'Reconcile summary '+key)
    need(summary['seed']==seed and summary['jobs']==48 and summary['cycles']==metrics[-1]['end_cycle'],'Summary identity')
    for left,right in zip(metrics,metrics[1:]):need(left['end_cycle']<right['admission_cycle'],'Release before next admission')
    return dict(status='PASS_QKV_DISPATCH_R1_TRACE',summary=summary,metrics=metrics,arithmetic_qualified=False,
        scope='Real append, cache guard and head sequencer; synthetic immutable source, AXI memory and arithmetic head. Independently reconstructed accepted bytes and ordering; no physical DDR/performance qualification.')

def review(text,seed):return review_parsed(*parse(text),seed)

def controls(text,seed):
    jobs,summary=parse(text);review_parsed(jobs,summary,seed);rejected=[]
    def reject(label,job,events):
        try:review_job(job,events)
        except (ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corruption '+label)
    changes=[(0,'BEGIN','sequence',1<<40),(0,'ADMIT','epoch',1<<40),(0,'ADMIT','lease',1),
        (0,'RAM','address',1),(0,'RAM','data',1),(0,'KV','index',1),(0,'KV','data',1),
        (0,'COMMIT','status',1),(0,'COMMIT','frames',1),(0,'AW','address',4096),(0,'AW','beats',1),
        (0,'W','index',1),(0,'W','data',1),(0,'B','response',2),(0,'B','beats',1),
        (0,'KV_ACK','bytes',64),(0,'KV_ACK','tokens',1),(0,'ROOT','prefix',1),
        (0,'Q','index',1),(0,'Q','data',1),(0,'HEAD','head',1),(0,'HEAD','query',1),
        (0,'HEAD_ACK','head',1),(0,'COMPLETE','status',1),(0,'COMPLETE','committed',1),
        (0,'COMPLETE','kv',1),(0,'COMPLETE','q',1),(0,'COMPLETE','bytes',64),
        (0,'RESULT','index',1),(0,'RESULT','data',1),(0,'RELEASE','source_acks',1),
        (0,'RELEASE','aw',1),(0,'RELEASE','w',1),(0,'RELEASE','b',1),
        (43,'INJECT','kv',1),(48,'INJECT','q',1),(54,'REVOKED','mode',1),(61,'RESET','aw',1),
        (35,'COMPLETE','bytes',64),(36,'COMPLETE','root',1),(16,'COMPLETE','status',1)]
    for mode,tag,key,mask in changes:
        job=PLAN.index(mode)+1;events=jobs[job].copy();i=next(i for i,(k,r) in enumerate(events) if k==tag)
        row=dict(events[i][1]);row[key]^=mask;events[i]=(tag,row);reject(f'{mode}-{tag}-{key}',job,events)
    for tag in ('ADMIT','APPEND','RAM','KV','COMMIT','AW','W','B','KV_ACK','ROOT','Q','HEAD','HEAD_ACK','COMPLETE','RESULT','ACK','SOURCE_ACK','RELEASE'):
        events=jobs[1];i=next(i for i,(k,r) in enumerate(events) if k==tag)
        reject('missing-'+tag,1,events[:i]+events[i+1:]);reject('duplicate-'+tag,1,events[:i]+[events[i]]+events[i:])
    for mode,tag in ((43,'INJECT'),(54,'REVOKED'),(61,'RESET')):
        job=PLAN.index(mode)+1;events=jobs[job];i=next(i for i,(k,r) in enumerate(events) if k==tag)
        reject('missing-'+tag,job,events[:i]+events[i+1:])
    for key in ('frames','heads','words','errors','revoked','resets','jobs','cycles','seed'):
        bad=dict(summary);bad[key]+=1
        try:review_parsed(jobs,bad,seed)
        except ValueError:rejected.append('summary-'+key)
        else:raise AssertionError('Accepted summary '+key)
    return rejected
