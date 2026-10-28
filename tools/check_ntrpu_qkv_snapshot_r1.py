"""Reconcile accepted Q/K words, AXI V bytes, publication and release ordering."""
import re

FAULTS=list(range(10,18))+list(range(20,35))+list(range(40,49))+list(range(50,70))
def status(mode):
    if 10<=mode<=17:return 1
    if 20<=mode<=27 or mode in (54,55):return 2
    if mode in (28,29,31,32,33,56,57,58,59):return 11
    if mode==30:return 5
    if mode==34 or 50<=mode<=53:return 7
    if 40<=mode<=43:return 3
    if mode in (44,45):return 4
    if 46<=mode<=48:return 8
    return 0
PLAN=[0,1,2]+[m for fault in FAULTS for m in (fault,0)]
SCHEMA={
    'BEGIN':'job mode sequence epoch layer position va vb vz owned',
    'ADMIT':'cycle job sequence epoch layer position expected vv vs ve vp sv ss se sp cp hs st beats rv cancel',
    'AR':'cycle job address beats','R':'cycle job address response rid last data',
    'Q':'cycle job index data','COMPLETE':'cycle job status valid ar r q stored beats frames',
    'FRAME':'cycle job index data','ACK':'cycle job',
    'SOURCE_ACK':'cycle job sequence epoch position','SOURCE_RESTORED':'cycle job',
    'CANCEL':'cycle job ar r stored','EPOCH':'cycle job ar r stored',
    'SOURCE_FAULT':'cycle job mode ar r stored','REVOKE':'cycle job mode',
    'INVALID_READ':'cycle job','RESET_BEGIN':'cycle job ar r stored',
    'RESET':'cycle job ar r q','RELEASE':'cycle job acks',
    'PASS':'seed jobs completions resets revocations ar r q frames acks ar_stalls r_stalls r_gaps cycles'}
HEX={'sequence','epoch','va','vb','vz','expected','vs','ve','ss','se','address','data'}

def need(ok,why):
    if not ok:raise ValueError(why)

def qword(index,job):return 0x3f000000|(((index*33+job*97)^0x345678)&0x7fffff)
def vword(address,job):return 0x3f000000|(((address>>2)^(address>>33)^(address>>40)^(job*71))&0x7fffff)

def parse(text):
    need(not re.search(r'(?m)^(FATAL:|ERROR:|FAIL\b|Fatal:)',text),'Failure diagnostic')
    jobs={};summary=None;cycle=0
    for line in text.splitlines():
        if not line.startswith('SNAP_'):continue
        kind,*tokens=line[5:].split();need(kind in SCHEMA,'Known event')
        fields=[t.split('=') for t in tokens]
        need(all(len(f)==2 for f in fields),'Field syntax')
        need([f[0] for f in fields]==SCHEMA[kind].split(),'Exact event fields')
        row={}
        for k,v in fields:
            width=128 if k=='data' and kind in ('R','FRAME') else 8 if k=='data' else 16
            need(re.fullmatch('[0-9a-f]{'+str(width)+'}' if k in HEX else '[0-9]+',v) is not None,'Canonical numeric field')
            row[k]=int(v,16 if k in HEX else 10)
        if 'cycle' in row:need(row['cycle']>=cycle,'Monotonic event clock');cycle=row['cycle']
        need(summary is None,'No event after summary')
        if kind=='PASS':summary=row;continue
        if kind=='BEGIN':
            need(row['job']==len(jobs)+1,'Ordered jobs');jobs[row['job']]=[]
        need(row['job']==len(jobs) and row['job'] in jobs,'Current job event')
        jobs[row['job']].append((kind,row))
    need(len(jobs)==len(PLAN)==107 and summary is not None,'Complete bank')
    return jobs,summary

def review_job(job,events):
    mode=PLAN[job-1];want=status(mode)
    need(events and events[0][0]=='BEGIN','Initial descriptor')
    desc=events[0][1];seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job
    layer=16 if mode==16 else 15 if mode==69 else job%16
    pos=131072 if mode==17 else 131071 if mode==69 else job*97
    offset=0 if mode in (1,42,43) else 128 if mode==2 else 4032
    base=0xc000000000;address=base+offset;size=16384
    if mode==10:address+=1
    if mode==11:size=offset+2047
    if mode==12:base=address+64
    if mode==13:size=0
    if mode==14:address=(1<<64)-64;base=0;size=(1<<64)-1
    if mode==15:address=base=(1<<64)-4096;size=8192
    owned=int(mode not in (24,25,26,28))
    need(desc==dict(job=job,mode=mode,sequence=seq,epoch=epoch,layer=layer,position=pos,va=address,vb=base,vz=size,owned=owned),'Complete command descriptor')
    by={k:[] for k in SCHEMA};pending=None;ars=[];rs=[];qs=[];frames=[]
    admit=None;complete=None;ack=None;reset_begin=None;reset=None;release=None;revoke=None;restore=None
    last_cycle=0
    for kind,row in events[1:]:
        need(kind in by and row.get('job')==job,'Event job identity')
        need('cycle' in row and row['cycle']>=last_cycle,'Per-job event order');last_cycle=row['cycle']
        need(release is None and reset is None,'No work after release/reset')
        by[kind].append(row)
        if kind=='ADMIT':
            need(admit is None and not ars and not qs,'Unique admission before work')
            expected=dict(cycle=row['cycle'],job=job,sequence=seq,epoch=epoch,layer=layer,position=pos,
                expected=epoch^((1<<40) if mode==23 else 0),vv=int(mode!=29),
                vs=seq^((1<<40) if mode==20 else 0),ve=epoch^((1<<40) if mode==21 else 0),vp=pos+int(mode==22),
                sv=int(mode!=28),ss=seq^((1<<40) if mode==24 else 0),se=epoch^((1<<40) if mode==25 else 0),
                sp=pos+int(mode==26),cp=pos+int(mode==27),hs=3 if mode==32 else 0,st=5 if mode==30 else 0,
                beats=163 if mode==33 else 164,rv=int(mode!=31),cancel=int(mode==34))
            need(row==expected,'Actual admitted source metadata');admit=row
        else:
            need(admit is not None and row['cycle']>=admit['cycle'],'No pre-admission action')
            if kind=='AR':
                need(complete is None and reset_begin is None and pending is None and not qs,'One outstanding V burst')
                need(row['cycle']>admit['cycle'],'Read after admission')
                a=address+len(rs)*64;n=min(32-len(rs),(4096-a%4096)//64)
                need(row['address']==a and row['beats']==n and n>0,'Page-bounded V extent')
                pending=dict(address=a,beats=n,index=0,cycle=row['cycle']);ars.append(row)
            elif kind=='R':
                need(pending is not None and complete is None and reset_begin is None,'Response belongs to issued burst')
                i=pending['index'];a=pending['address']+i*64
                need(row['cycle']>pending['cycle'] and row['address']==a,'Ordered R address')
                last=i==(3 if mode==42 else pending['beats'] if mode==43 else pending['beats']-1)
                need(row['last']==int(last),'Expected RLAST boundary')
                need(row['response']==(2 if mode==40 and a==address+1984 else 0),'RRESP stimulus')
                need(row['rid']==int(mode==41 and a==address+1984),'RID stimulus')
                data=sum(vword(a+lane*4,job)<<(lane*32) for lane in range(16))
                if mode==44 and a==address+1984:data=(data&((1<<480)-1))|(0x7fc12345<<480)
                need(row['data']==data,'Actual V memory bytes');rs.append(row);pending['index']+=1
                if last:pending=None
            elif kind=='Q':
                need(complete is None and reset_begin is None and pending is None and len(rs)==32,'Q/K after V drained')
                need(row['index']==len(qs)<2560,'Sequential source RAM')
                need(row['data']==(0x7fc12345 if mode==45 and len(qs)==2559 else qword(len(qs),job)),'Source Q/K word')
                qs.append(row)
            elif kind in ('CANCEL','EPOCH','SOURCE_FAULT','RESET_BEGIN'):
                need(row['cycle']>admit['cycle'],'Fault/reset after actual admission')
                need(row['ar']==len(ars) and row['r']==len(rs) and row['stored']<=len(qs),'Fault progress reflects this job')
                if kind=='RESET_BEGIN':need(reset_begin is None,'Unique reset');reset_begin=row
                else:need(complete is None,'Mid-flight fault before completion')
                if mode in (53,55,56,57,58,59,66):need(row['stored']==63 and len(qs)==63 and len(rs)==32,'Mid-copy fault exercised')
                if mode in (52,54):need(len(rs)==8 and not qs,'Fault during accepted V burst')
                if mode in (50,51,64):need(not ars and not rs and not qs,'Pre-AR fault/reset')
                if mode==65:need(len(ars)==1 and not rs and not qs,'Reset with R outstanding')
                if kind=='SOURCE_FAULT':need(row['mode']==mode,'Source fault kind')
            elif kind=='COMPLETE':
                need(complete is None and reset_begin is None and pending is None,'Unique drained completion')
                need(row['status']==want and row['valid']==int(want==0),'Completion status/publication')
                need((row['ar'],row['r'],row['q'])==(len(ars),len(rs),len(qs)),'Completion transfer accounting')
                need(0<=row['stored']<=len(qs)<=2560 and 0<=row['beats']<=len(rs),'Private extents bounded')
                need(row['frames']==(192 if want==0 else 0),'Public frame extent')
                if want==0:need(len(rs)==row['beats']==32 and len(qs)==row['stored']==2560,'Full accepted snapshot')
                if 10<=mode<=34:need(not ars and not qs and not rs,'Admission rejects before work')
                if mode==45:need(row['stored']==2559 and len(qs)==2560,'Reject final nonfinite Q/K word')
                if mode in (46,47,48):need(not qs,'Timeout did not publish Q/K')
                complete=row
            elif kind=='FRAME':
                need(complete is not None and want==0 and ack is None and revoke is None and reset_begin is None,'Authorized public frame')
                index=(len(frames)*17+31)%192
                need(len(frames)<192 and row['index']==index and row['cycle']>complete['cycle'],'Full permuted readback')
                if index<160:data=sum(qs[index*16+lane]['data']<<(lane*32) for lane in range(16))
                else:data=rs[index-160]['data']
                need(row['data']==data,'Packed frame equals accepted source bytes');frames.append(row)
            elif kind=='INVALID_READ':
                need(mode==68 and len(frames)==192 and ack is None,'Invalid frame bounds exercised')
            elif kind=='REVOKE':
                need(mode in (60,61,62,63) and row['mode']==mode and complete is not None and len(frames)==192 and ack is None and revoke is None,'Held-result revocation')
                revoke=row
            elif kind=='ACK':
                need(complete is not None and ack is None and reset_begin is None,'Public ACK after completion')
                need(row['cycle']>=complete['cycle']+8,'Held completion before ACK')
                if want==0:need(len(frames)==192,'Consumer finished readback')
                ack=row
            elif kind=='SOURCE_RESTORED':
                need(mode in (57,63) and ack is not None and row['cycle']>=ack['cycle']+5 and not by['SOURCE_ACK'],'Foreign identity retained until restored')
                need(restore is None,'Unique restored identity');restore=row
            elif kind=='SOURCE_ACK':
                need(owned and ack is not None and row['cycle']>ack['cycle'],'No early/foreign source ACK')
                need(row==dict(cycle=row['cycle'],job=job,sequence=seq,epoch=epoch,position=pos),'Full release identity')
                if mode in (57,63):need(restore is not None and row['cycle']>restore['cycle'],'Source restored before ACK')
                need(len(by[kind])==1,'Single source ACK')
            elif kind=='RELEASE':
                need(ack is not None and row['acks']==len(by['SOURCE_ACK'])==owned,'Final release accounting')
                need(row['cycle']>max(ack['cycle'],by['SOURCE_ACK'][-1]['cycle'] if owned else 0),'Reuse after source drain')
                release=row
            elif kind=='RESET':
                need(reset_begin is not None and row['cycle']>=reset_begin['cycle']+2 and ack is None and not by['SOURCE_ACK'],'Coordinated reset without ACK')
                need((row['ar'],row['r'],row['q'])==(len(ars),len(rs),len(qs)),'Reset accounting');reset=row
            else:raise ValueError('Unexpected event '+kind)
    need(admit is not None,'Command accepted')
    for tag,required in [('CANCEL',50<=mode<=53),('EPOCH',mode in (54,55)),('SOURCE_FAULT',56<=mode<=59),
        ('REVOKE',60<=mode<=63),('RESET_BEGIN',64<=mode<=67),('RESET',64<=mode<=67),
        ('INVALID_READ',mode==68),('SOURCE_RESTORED',mode in (57,63))]:
        need(len(by[tag])==int(required),'Exact scenario coverage '+tag)
    if 64<=mode<=67:
        need(reset is not None and release is None and (complete is not None)==(mode==67),'Reset scenario terminal')
    else:need(complete is not None and release is not None and reset is None,'Completion/release scenario terminal')
    need(len(frames)==(192 if complete is not None and want==0 else 0),'All and only valid frames')
    return dict(job=job,mode=mode,ar=len(ars),r=len(rs),q=len(qs),frames=len(frames),acks=len(by['SOURCE_ACK']),
        completions=int(complete is not None),resets=int(reset is not None),revocations=int(revoke is not None),
        admission_cycle=admit['cycle'],completion_cycle=complete['cycle'] if complete else None,end_cycle=last_cycle)

def review_parsed(jobs,summary,seed):
    need(list(jobs)==list(range(1,108)),'Exact job identities')
    metrics=[review_job(j,events) for j,events in jobs.items()]
    for k in ('ar','r','q','frames','acks','completions','resets','revocations'):
        need(summary[k]==sum(row[k] for row in metrics),'Summary reconciles '+k)
    need(summary['seed']==seed and summary['jobs']==107 and summary['cycles']==metrics[-1]['end_cycle'],'Bank summary')
    need(summary['ar_stalls']>0 and summary['r_gaps']>0 and summary['r_stalls']>=0,'Channel coverage')
    for left,right in zip(metrics,metrics[1:]):need(left['end_cycle']<right['admission_cycle'],'Job release before reuse')
    return dict(status='PASS_QKV_SNAPSHOT_R1_TRACE',summary=summary,jobs=107,metrics=metrics,
        verified_frame_bytes=summary['frames']*64,arithmetic_qualified=False,
        scope='Synthetic immutable Q/K source and AXI V memory. Every public frame reconstructed from accepted words/beats; identities, gates and bounds also asserted by the compiled bank. No downstream KV publication or inference.')

def review(text,seed):return review_parsed(*parse(text),seed)

def controls(text,seed):
    jobs,summary=parse(text);review_parsed(jobs,summary,seed);rejected=[]
    def reject_job(label,j,events):
        try:review_job(j,events)
        except (ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corruption '+label)
    changes=[(1,'BEGIN','sequence',1<<40),(1,'BEGIN','va',64),(1,'ADMIT','ss',1<<40),
        (1,'ADMIT','se',1<<40),(1,'ADMIT','cp',1),(1,'AR','address',4096),(1,'AR','beats',1),
        (1,'R','data',1),(1,'R','rid',1),(1,'R','last',1),(1,'Q','index',1),(1,'Q','data',1),
        (1,'COMPLETE','valid',1),(1,'COMPLETE','stored',1),(1,'COMPLETE','frames',1),
        (1,'FRAME','data',1),(1,'FRAME','index',1),(1,'SOURCE_ACK','sequence',1<<40),
        (1,'SOURCE_ACK','epoch',1<<40),(1,'SOURCE_ACK','position',1),(1,'RELEASE','acks',1)]
    for mode,tag,key,mask in [(53,'CANCEL','stored',64),(55,'EPOCH','stored',64),(57,'SOURCE_FAULT','stored',64),
        (63,'REVOKE','mode',1),(66,'RESET_BEGIN','stored',64),(65,'RESET','ar',1)]:
        changes.append((PLAN.index(mode)+1,tag,key,mask))
    for j,tag,key,mask in changes:
        events=jobs[j].copy();i=next(i for i,(k,r) in enumerate(events) if k==tag)
        row=dict(events[i][1]);row[key]^=mask;events[i]=(tag,row);reject_job(f'{j}-{tag}-{key}',j,events)
    for tag in ('ADMIT','AR','R','Q','COMPLETE','FRAME','ACK','SOURCE_ACK','RELEASE'):
        events=jobs[1];i=next(i for i,(k,r) in enumerate(events) if k==tag)
        reject_job('missing-'+tag,1,events[:i]+events[i+1:])
        reject_job('duplicate-'+tag,1,events[:i]+[events[i]]+events[i:])
    for mode,tag in [(57,'SOURCE_RESTORED'),(66,'RESET_BEGIN'),(63,'REVOKE'),(68,'INVALID_READ')]:
        j=PLAN.index(mode)+1;events=jobs[j];i=next(i for i,(k,r) in enumerate(events) if k==tag)
        reject_job('missing-'+tag,j,events[:i]+events[i+1:])
    for tag,before in [('SOURCE_ACK','ACK'),('COMPLETE','Q'),('FRAME','COMPLETE')]:
        events=jobs[1].copy();i=next(i for i,(k,r) in enumerate(events) if k==tag);event=events.pop(i)
        at=next(i for i,(k,r) in enumerate(events) if k==before);events.insert(at,event)
        reject_job('premature-'+tag,1,events)
    for key in ('r','q','frames','acks','completions','resets','revocations'):
        bad=dict(summary);bad[key]+=1
        try:review_parsed(jobs,bad,seed)
        except ValueError:rejected.append('summary-'+key)
        else:raise AssertionError('Accepted summary '+key)
    need(len(rejected)==59,'Control inventory')
    return rejected
