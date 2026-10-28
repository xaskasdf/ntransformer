"""Reconstruct the real capture/dispatch path from accepted external data."""
import re
from check_ntrpu_qkv_dispatch_r2 import need,word

PLAN=[0,3,1,2,10,11,20,21,24,25,26,28,30,31,32,33,34,35,40,43,50,51,52,53,*range(80,89),90,91]
RESETS=(50,51,52,53,90,91)
SCHEMA={
 'CAPTURE_BEGIN':'job sequence epoch position va',
 'DISPATCH_BEGIN':'job mode sequence epoch layer position base capacity',
 'DISPATCH_ADMIT':'cycle job sequence epoch layer position lease',
 'CAPTURE_AR':'cycle job address beats','CAPTURE_R':'cycle job address response rid last data',
 'CAPTURE_QK':'cycle job index data','CAPTURE_QK_RETURN':'cycle job index data',
 'CAPTURE_COMPLETE':'cycle job status qk v frames','CAPTURE_SOURCE_ACK':'cycle job sequence epoch position',
 'CAPTURE_INJECT':'cycle job mode qk v kv q','CAPTURE_RESET':'cycle job qk v kv q heads',
 'CAPTURE_REVOKED':'cycle job mode','CAPTURE_SOURCE_RESTORED':'cycle job',
 'DISPATCH_APPEND':'cycle job','DISPATCH_RAM':'cycle job address data',
 'DISPATCH_KV':'cycle job index data','DISPATCH_COMMIT':'cycle job status frames',
 'DISPATCH_AW':'cycle job address beats','DISPATCH_W':'cycle job index data','DISPATCH_B':'cycle job beats response',
 'DISPATCH_KV_ACK':'cycle job status bytes tokens','DISPATCH_ROOT':'cycle job prefix',
 'DISPATCH_Q':'cycle job index data','DISPATCH_HEAD':'cycle job head query','DISPATCH_HEAD_ACK':'cycle job head',
 'CAPDISP_COMPLETE':'cycle job status capture append root qk v kv q bytes tokens committed',
 'DISPATCH_RESULT':'cycle job index data','DISPATCH_ACK':'cycle job','DISPATCH_CAPTURE_ACK':'cycle job',
 'CAPDISP_RELEASE':'cycle job aw w b capture_acks rope_acks reset',
 'ROPECAP_AR':'cycle job plane address beats',
 'ROPECAP_R':'cycle job address response data',
 'ROPECAP_HSTART':'cycle job','ROPECAP_HDONE':'cycle job reads writes',
 'ROPECAP_WRITE':'cycle job index data checked','ROPECAP_COMPLETE':'cycle job status beats hls',
 'ROPECAP_HANDOFF':'cycle job sequence epoch layer position lease',
 'ROPECAP_INJECT':'cycle job mode r reads writes',
 'ROPECAP_PROTOCOL_PASS':'seed jobs frames heads words cycles qk v acks errors resets revoked'}

def status(mode):
    if mode in (80,81,82,83):return 7
    if mode in (84,88):return 2
    if mode==85:return 3
    if mode==86:return 4
    if mode==87:return 1
    if mode==10:return 1
    if mode in (11,26):return 2
    if mode in (12,35):return 5
    if mode in (13,27):return 11
    if mode in (20,34):return 3
    if mode in (21,22):return 4
    if mode==23:return 8
    if mode in (24,25,28,30,31,32,33):return 7
    return 0

def parse(text):
    need(not re.search(r'(?m)^(FATAL:|Fatal:|ERROR:|FAIL\b)',text),'Failure diagnostic')
    jobs={};summary=None;cycle=0
    for line in text.splitlines():
        if not line.startswith(('CAPTURE_','DISPATCH_','CAPDISP_','ROPECAP_')):continue
        tag,*tokens=line.split();need(tag in SCHEMA,'Known event')
        fields=[t.split('=') for t in tokens]
        need(all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split(),'Exact fields')
        row={}
        for key,val in fields:
            width=512 if key=='query' else 128 if key=='data' and tag in ('CAPTURE_R','DISPATCH_RAM','DISPATCH_KV','DISPATCH_Q','DISPATCH_W','ROPECAP_R') else 8 if key=='data' else 16 if key in ('sequence','epoch','base','lease','va') or (key=='address' and tag in ('CAPTURE_AR','CAPTURE_R','DISPATCH_AW','ROPECAP_AR','ROPECAP_R')) else 0
            need(re.fullmatch('[0-9a-f]{'+str(width)+'}' if width else '[0-9]+',val) is not None,'Numeric shape');row[key]=int(val,16 if width else 10)
        if 'cycle' in row:need(row['cycle']>=cycle,'Monotonic event time');cycle=row['cycle']
        need(summary is None,'No events after summary')
        if tag=='ROPECAP_PROTOCOL_PASS':summary=row;continue
        if tag=='CAPTURE_BEGIN':need(row['job']==len(jobs)+1,'Sequential jobs');jobs[row['job']]=[]
        need(row['job']==len(jobs) and row['job'] in jobs,'Current job event');jobs[row['job']].append((tag,row))
    need(len(jobs)==35 and summary is not None,'Complete bank')
    return jobs,summary

def review_downstream(job,events):
    mode=PLAN[job-1];want=status(mode);seq=0x1234567800000000+(job-1 if mode==3 else job)
    epoch=0xfedcba9800000000+(job-1 if mode==3 else job);pos=int(mode==3);layer=15 if mode==1 else 0;cap=3 if mode==1 else 2
    base=0x900000000fc0;va=0xc000000000+(0 if mode==1 else 128 if mode==2 else 4032)
    lease=0xffffffffffffffe0 if mode==2 else 0x2000000000001000+job*64
    need(events[0]==('CAPTURE_BEGIN',dict(job=job,sequence=seq,epoch=epoch,position=pos,va=va)),'Capture descriptor')
    need(events[1]==('DISPATCH_BEGIN',dict(job=job,mode=mode,sequence=seq,epoch=epoch,layer=layer,position=pos,base=base,capacity=cap)),'Dispatch descriptor')
    by={tag:[] for tag in SCHEMA};last=0
    for tag,row in events[2:]:
        need(row['job']==job and row['cycle']>=last,'Ordered job events');last=row['cycle'];by[tag].append(row)
    def one(tag,required=True):
        need(len(by[tag])==int(required),'Unique/required '+tag);return by[tag][0] if required else None
    admit=one('DISPATCH_ADMIT');release=one('CAPDISP_RELEASE');reset=one('CAPTURE_RESET',mode in RESETS)
    cc=one('CAPTURE_COMPLETE',mode not in (50,51,90,91));complete=one('CAPDISP_COMPLETE',mode not in RESETS)
    ack=one('DISPATCH_ACK',mode not in RESETS);ca=one('DISPATCH_CAPTURE_ACK',mode not in RESETS);sa=one('CAPTURE_SOURCE_ACK',mode not in RESETS)
    inject=one('CAPTURE_INJECT',24<=mode<=28 or 30<=mode<=33)
    revoke=one('CAPTURE_REVOKED',40<=mode<=43);restore=one('CAPTURE_SOURCE_RESTORED',mode in (27,42))
    need(admit==dict(cycle=admit['cycle'],job=job,sequence=seq,epoch=epoch,layer=layer,position=pos,lease=lease),'Actual accepted descriptor')
    need(events[-1][0]=='CAPDISP_RELEASE' and last==release['cycle'],'Final release event')
    for tag,rows in by.items():
        if tag in ('CAPTURE_BEGIN','DISPATCH_BEGIN','DISPATCH_ADMIT'):continue
        for r in rows:
            need(r['cycle']>=admit['cycle'] and r['cycle']<=release['cycle'],'Events inside command lifetime')
            if reset and tag!='CAPDISP_RELEASE':need(r['cycle']<=reset['cycle'],'No work after reset')
    # Accepted V bus: burst sizes, response identity, actual payload and drain.
    pending=None;rs=[];requests=[];returns=[];source_pending=None
    for tag,r in events[2:]:
        if tag=='CAPTURE_AR':
            need(pending is None and len(rs)<32 and not requests and r['cycle']>admit['cycle'],'One V burst before Q/K')
            address=va+len(rs)*64;beats=min(32-len(rs),(4096-address%4096)//64)
            need(r['address']==address and r['beats']==beats,'V page-bounded allocation');pending=dict(address=address,beats=beats,index=0,cycle=r['cycle'])
        elif tag=='CAPTURE_R':
            need(pending is not None,'V response owns request');index=len(rs);a=pending['address']+pending['index']*64
            need(r['address']==a and r['cycle']>pending['cycle'],'Ordered V response')
            need(r['response']==(2 if mode==20 and index==31 else 0) and r['rid']==0 and r['last']==int(pending['index']==pending['beats']-1),'V response attributes')
            data=sum(word(2560+index*16+i,job)<<(i*32) for i in range(16))
            if mode==21 and index==31:data=(data&~0xffffffff)|0x7fc00001
            need(r['data']==data,'Actual V memory bytes');rs.append(r);pending['index']+=1
            if r['last']:pending=None
        elif tag=='CAPTURE_QK':
            need(pending is None and len(rs)==32 and source_pending is None,'RoPE read after V drain')
            need(r['index']==len(requests)<2560 and r['data']==word(len(requests),job),'Ordered RoPE request');requests.append(r);source_pending=r
        elif tag=='CAPTURE_QK_RETURN':
            need(source_pending is not None and r['cycle']>source_pending['cycle'] and r['index']==source_pending['index'],'Matched synchronous RAM response')
            need(r['data']==(0x7fc00001 if mode==22 and r['index']==2048 else word(r['index'],job)),'Actual returned RoPE word')
            returns.append(r);source_pending=None
    if not reset:need(pending is None,'Drain V before completion')
    if cc:
        cap_status=want if 10<=mode<=27 or 80<=mode<=88 else 0
        need(cc['status']==cap_status,'Capture preserves its own first fault')
        stored=[]
        for r in returns:
            if ((r['data']>>23)&255)==255 or (inject and mode in (24,25,26,27) and r['cycle']>inject['cycle']):break
            stored.append(r)
        need(cc['qk']==len(stored) and cc['v']==len(rs)-int(mode==20),'Actual capture write extents')
        need(cc['frames']==(192 if cap_status==0 and mode!=28 else 0),'Capture publication extent')
        if cap_status==0:need(len(returns)==2560 and len(rs)==32,'Complete source before handoff')
        if mode in (10,11,12,13) or 80<=mode<=88:need(not rs and not requests and not by['CAPTURE_AR'],'Invalid capture command has no reads')
        if mode in (20,21,24):need(not requests,'V failure prevents Q/K reads')
        if mode==22:need(len(requests)==len(returns)==2049 and cc['qk']==2048,'Nonfinite Q/K stops capture')
        if mode==23:need(len(requests)==2560 and len(returns)==2559 and source_pending is not None,'Missing final RAM response times out')
    captured={i:r['data'] for i,r in enumerate(returns)}
    def captured_frame(index):
        if index<160:
            need(all(index*16+i in captured for i in range(16)),'Frame owns sixteen returned RoPE words')
            return sum(captured[index*16+i]<<(32*i) for i in range(16))
        need(0<=index-160<len(rs),'Frame owns accepted V beat');return rs[index-160]['data']
    kv=[];qs=[];ram=None;app=None;commit=None;root=None;kvack=None
    aws=[];ws=[];bs=[];pending=None;burst=[];planned=0;committed={};heads=[];hacks=[];results=[]
    location=lambda beat:base+((layer*16+beat//4)*cap+pos)*256+(beat%4)*64
    for tag,r in events[2:]:
        if tag=='DISPATCH_APPEND':
            need(app is None and cc is not None and cc['status']==0 and cc['frames']==192 and r['cycle']>cc['cycle'],'Append only after full real capture');app=r
        elif tag=='DISPATCH_RAM':
            index=len(qs) if root else 128+len(kv)
            need(app is not None and ram is None and r['address']==index and r['data']==captured_frame(index),'Actual captured frame address and bytes');ram=r
        elif tag in ('DISPATCH_KV','DISPATCH_Q'):
            vector=kv if tag=='DISPATCH_KV' else qs;offset=128 if tag=='DISPATCH_KV' else 0
            need((root is not None)==(tag=='DISPATCH_Q') and r['index']==len(vector)<(64 if offset else 128),'Ordered KV/Q phase')
            need(ram is not None and ram['address']==offset+len(vector) and r['cycle']>ram['cycle'] and r['data']==ram['data']==captured_frame(ram['address']),'Transferred bytes equal retained capture')
            need(all(((r['data']>>(32*i+23))&255)!=255 for i in range(16)),'Finite downstream frame');vector.append(r);ram=None
        elif tag=='DISPATCH_COMMIT':
            need(app is not None and commit is None and not aws and not ws and r['frames']==len(kv),'Unique producer commit before writes')
            need(r['status'] in (0,want) and (r['status']!=0 or len(kv)==64),'Commit completeness');commit=r
        elif tag=='DISPATCH_AW':
            need(commit is not None and commit['status']==0 and pending is None and root is None,'Write after successful producer commit')
            a=location(planned);n=min(4-planned%4,(4096-a%4096)//64)
            need(r['address']==a and r['beats']==n and planned+n<=64,'Physical KV layout')
            pending=dict(start=planned,beats=n,cycle=r['cycle']);planned+=n;aws.append(r)
        elif tag=='DISPATCH_W':
            need(commit is not None and commit['status']==0 and r['index']==len(ws)<len(kv) and len(burst)<4,'Write channel ownership')
            need(r['data']==kv[len(ws)]['data'],'DDR write equals captured KV');ws.append(r);burst.append(r)
        elif tag=='DISPATCH_B':
            need(pending is not None and r['beats']==pending['beats']==len(burst) and r['cycle']>pending['cycle'],'Complete write burst before B')
            need([r['index'] for r in burst]==list(range(pending['start'],planned)),'Address/data correspondence')
            need(r['response']==(2 if mode==34 and planned==64 else 0),'B stimulus')
            if r['response']==0:committed.update({r['index']:r['data'] for r in burst})
            pending=None;burst=[];bs.append(r)
        elif tag=='DISPATCH_KV_ACK':
            need(kvack is None and app is not None and pending is None and not burst and len(aws)==len(bs),'Append response after drain')
            need(r['bytes']==len(ws)*64 and r['tokens']==pos+int(r['status']==0),'W bytes and published prefix')
            if r['status']==0:need(len(committed)==len(kv)==64 and r['bytes']==4096,'Full successful append')
            else:need(r['status'] in (3,7),'Expected append failure')
            kvack=r
        elif tag=='DISPATCH_ROOT':
            need(root is None and kvack is not None and kvack['status']==0 and r['cycle']>kvack['cycle'] and r['prefix']==pos+1,'Root after acknowledged prefix')
            need(len(committed)==64 and all(committed[i]==captured_frame(128+i) for i in range(64)),'Full KV confirmed before attention');root=r
        elif tag=='DISPATCH_HEAD':
            need(root is not None and len(qs)==128 and len(heads)==len(hacks) and r['head']==len(heads)<32,'Ordered owned head')
            need(r['query']==sum(captured[len(heads)*64+i]<<(32*i) for i in range(64)),'Actual head query from RoPE capture');heads.append(r)
        elif tag=='DISPATCH_HEAD_ACK':
            need(len(heads)==len(hacks)+1 and r['head']==len(hacks) and r['cycle']>heads[-1]['cycle'],'Drain owned head');hacks.append(r)
        elif tag=='DISPATCH_RESULT':
            need(complete is not None and want==0 and r['cycle']>complete['cycle'] and r['cycle']<ack['cycle'] and len(results)<2048,'Public authorized result')
            index=(len(results)*17+31)%2048
            need(r['index']==index and r['data']==(((heads[index//64]['query']>>(32*(index%64)))&0xffffffff)^0x100),'Result reconstructed from accepted head query');results.append(r)
    if complete:
        need(pending is None and not burst and len(heads)==len(hacks),'Completion after memory/head drain')
        need(complete['status']==want and complete['capture']==cc['status'] and complete['qk']==cc['qk'] and complete['v']==cc['v'],'Composed statuses and capture extent')
        need((complete['kv'],complete['q'])==(len(kv),len(qs)),'Transferred dispatch extent')
        need((complete['append'],complete['bytes'],complete['tokens'])==((kvack['status'],kvack['bytes'],kvack['tokens']) if kvack else (0,0,0)),'Retained append reply')
        need(complete['committed']==int(kvack is not None and kvack['status']==0),'Preserve successful KV commit')
        need(complete['root']==(7 if mode==32 else 5 if mode==35 else 0),'Root failure provenance')
        if 10<=mode<=28 or 80<=mode<=88:need(app is None and not kv and root is None,'Capture failure prevents side effects')
        if want==0:need(len(kv)==64 and len(qs)==128 and len(heads)==32 and complete['committed'],'Whole success')
        need(ack['cycle']>=complete['cycle']+6 and ca['cycle']>ack['cycle'] and sa['cycle']>ca['cycle'] and release['cycle']>sa['cycle'],'Public then capture then RoPE release')
        need(sa==dict(cycle=sa['cycle'],job=job,sequence=seq,epoch=epoch,position=pos),'Full upstream release identity')
        need(len(results)==(2048 if want==0 else 0),'Exact public extent')
    if inject:
        need(inject['mode']==mode,'Fault kind')
        seen=lambda tag:sum(r['cycle']<=inject['cycle'] for r in by[tag])
        need((inject['qk'],inject['v'],inject['kv'],inject['q'])==(seen('CAPTURE_QK'),seen('CAPTURE_R'),seen('DISPATCH_KV'),seen('DISPATCH_Q')),'Fault progress')
        if mode in (25,26,27):need(inject['qk']==2048,'Fault during actual Q/K capture')
        if mode==24:need(inject['v']==8,'Fault with V burst active')
        if mode==28:need(cc['status']==0 and cc['frames']==0 and complete['status']==7 and inject['cycle']<cc['cycle'],'Cancel pulse retained across capture/dispatch handoff')
        if mode==30:need(inject['kv']==17,'Fault during KV stream')
        if mode==32:need(inject['q']==17,'Fault during Q stream')
        if mode==33:need(kvack is not None and inject['cycle']>=kvack['cycle'] and root is None and complete['committed'],'Cancel after KV commit')
    if restore:need(restore['cycle']>=ack['cycle']+7 and sa['cycle']>restore['cycle'],'Foreign upstream retained until restored')
    if revoke:need(revoke['mode']==mode and revoke['cycle']>results[-1]['cycle'] and revoke['cycle']<ack['cycle'],'Held result revoked before release')
    if reset:
        need((reset['qk'],reset['v'],reset['kv'],reset['q'],reset['heads'])==(len(requests),len(rs),len(kv),len(qs),len(heads)),'Reset progress')
        need(release['cycle']>reset['cycle'] and not results,'Reset releases without ACK/readback')
        if mode==50:need(len(rs)==8,'Reset during V')
        if mode==51:need(len(requests)==2048,'Reset during Q/K')
        if mode==52:need(len(qs)==17,'Reset during Q stream')
        if mode==53:need(len(heads)==len(hacks)==32,'Reset after all heads')
    need((release['aw'],release['w'],release['b'],release['capture_acks'],release['rope_acks'],release['reset'])==(len(aws),len(ws),len(bs),int(mode not in RESETS),int(mode not in RESETS),int(mode in RESETS)),'Release counters')
    return dict(job=job,mode=mode,frames=len(kv)+len(qs),heads=len(heads),words=len(results),qk=len(requests),v=len(rs),acks=int(mode not in RESETS),
        errors=int(want!=0),resets=int(mode in RESETS),revoked=int(revoke is not None),admission_cycle=admit['cycle'],end_cycle=release['cycle'])

def review_parsed(jobs,summary,seed):
    need(list(jobs)==list(range(1,36)),'Job inventory');metrics=[review_job(j,e) for j,e in jobs.items()]
    for key in ('frames','heads','words','qk','v','acks','errors','resets','revoked'):need(summary[key]==sum(m[key] for m in metrics),'Summary '+key)
    need(summary['seed']==seed and summary['jobs']==35 and summary['cycles']==metrics[-1]['end_cycle'],'Summary identity')
    for left,right in zip(metrics,metrics[1:]):need(left['end_cycle']<right['admission_cycle'],'No overlap on reuse')
    return dict(status='PASS_ROPE_QKV_DISPATCH_R1_PROTOCOL_TRACE',summary=summary,metrics=metrics,arithmetic_qualified=False)

def review(text,seed):return review_parsed(*parse(text),seed)

def controls(text,seed):
    jobs,summary=parse(text);review_parsed(jobs,summary,seed);rejected=[]
    def reject(label,j,events):
        try:review_job(j,events)
        except (ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corruption '+label)
    fields=[('ROPECAP_AR','address'),('ROPECAP_AR','beats'),('ROPECAP_R','data'),('ROPECAP_R','response'),('ROPECAP_WRITE','data'),('ROPECAP_WRITE','checked'),('ROPECAP_COMPLETE','beats'),('ROPECAP_HANDOFF','lease'),('CAPTURE_BEGIN','sequence'),('DISPATCH_BEGIN','layer'),('DISPATCH_ADMIT','lease'),
        ('CAPTURE_AR','address'),('CAPTURE_AR','beats'),('CAPTURE_R','data'),('CAPTURE_R','response'),('CAPTURE_R','last'),
        ('CAPTURE_QK','index'),('CAPTURE_QK_RETURN','data'),('CAPTURE_COMPLETE','qk'),('CAPTURE_COMPLETE','frames'),
        ('DISPATCH_RAM','data'),('DISPATCH_KV','data'),('DISPATCH_COMMIT','frames'),('DISPATCH_AW','address'),
        ('DISPATCH_W','data'),('DISPATCH_B','response'),('DISPATCH_KV_ACK','tokens'),('DISPATCH_ROOT','prefix'),
        ('DISPATCH_Q','data'),('DISPATCH_HEAD','query'),('CAPDISP_COMPLETE','committed'),('DISPATCH_RESULT','data'),
        ('CAPTURE_SOURCE_ACK','epoch'),('CAPDISP_RELEASE','rope_acks')]
    changes=[(0,tag,key) for tag,key in fields]+[(28,'CAPTURE_INJECT','qk'),(33,'CAPDISP_COMPLETE','committed'),(43,'CAPTURE_REVOKED','mode'),(50,'CAPTURE_RESET','v')]
    for mode,tag,key in changes:
        j=PLAN.index(mode)+1;events=jobs[j].copy();i=next(i for i,(t,r) in enumerate(events) if t==tag);row=dict(events[i][1]);row[key]^=1;events[i]=(tag,row);reject(f'{mode}-{tag}-{key}',j,events)
    for tag in ('ROPECAP_AR','ROPECAP_R','ROPECAP_HSTART','ROPECAP_HDONE','ROPECAP_WRITE','ROPECAP_COMPLETE','ROPECAP_HANDOFF','DISPATCH_ADMIT','CAPTURE_AR','CAPTURE_R','CAPTURE_QK','CAPTURE_QK_RETURN','CAPTURE_COMPLETE','DISPATCH_APPEND','DISPATCH_KV','DISPATCH_AW','DISPATCH_W','DISPATCH_B','DISPATCH_KV_ACK','DISPATCH_ROOT','DISPATCH_Q','DISPATCH_HEAD','DISPATCH_HEAD_ACK','CAPDISP_COMPLETE','DISPATCH_RESULT','DISPATCH_ACK','DISPATCH_CAPTURE_ACK','CAPTURE_SOURCE_ACK','CAPDISP_RELEASE'):
        events=jobs[1];i=next(i for i,(t,r) in enumerate(events) if t==tag)
        reject('missing-'+tag,1,events[:i]+events[i+1:]);reject('duplicate-'+tag,1,events[:i]+[events[i]]+events[i:])
    for key in summary:
        bad=dict(summary);bad[key]+=1
        try:review_parsed(jobs,bad,seed)
        except ValueError:rejected.append('summary-'+key)
        else:raise AssertionError('Accepted summary '+key)
    return rejected

def review_job(job,events):
    metrics=review_downstream(job,events);mode=PLAN[job-1]
    by={t:[r for tag,r in events if tag==t] for t in SCHEMA}
    def one(tag,present=True):
        need(len(by[tag])==int(present),'Required upstream event '+tag)
        return by[tag][0] if present else None
    reset=mode in RESETS;early_reset=mode in (90,91)
    admit=one('DISPATCH_ADMIT');done=one('ROPECAP_COMPLETE',not early_reset)
    handoff=one('ROPECAP_HANDOFF',not early_reset)
    inject=one('ROPECAP_INJECT',80<=mode<=84)
    starts=by['ROPECAP_HSTART'];ends=by['ROPECAP_HDONE'];writes=by['ROPECAP_WRITE']
    need(len(starts)<=1 and len(ends)<=1,'Unique protocol core lifetime')
    if not early_reset:need(len(starts)==len(ends),'Core drains before response')
    pending=None;rs=[];plane=0;index=0;memory={};addresses=(0x100000fc0,0x8000000080,0xa000000000);extents=(128,32,4)
    for tag,r in events:
        if tag=='ROPECAP_AR':
            need(pending is None and plane<3 and r['cycle']>admit['cycle'],'Upstream owned AXI request')
            a=addresses[plane]+64*index;n=min(extents[plane]-index,(4096-a%4096)//64)
            need(r['address']==a and r['beats']==n and r['plane']==plane,'Q/K/phase allocation and page bounds')
            pending=dict(address=a,beats=n,index=0,plane=plane,cycle=r['cycle'])
            index+=n
            if index==extents[plane]:index=0;plane+=1
        elif tag=='ROPECAP_R':
            need(pending is not None and r['cycle']>pending['cycle'],'Upstream response owns burst')
            a=pending['address']+pending['index']*64;p=pending['plane']
            need(r['address']==a and r['response']==(2 if mode==85 and len(rs)==163 else 0),'Upstream ordered response')
            first=(a-addresses[p])//4+(2048 if p==1 else 2560 if p==2 else 0)
            data=sum((word(first+i,job) if p<2 else 0x3f800000)<<(32*i) for i in range(16))
            if mode==86 and len(rs)==163:data=(data&~0xffffffff)|0x7fc00001
            need(r['data']==data,'Actual Q/K/phase bus bytes')
            if r['response']==0:
                for i in range(16):memory[first+i]=(r['data']>>(32*i))&0xffffffff
            rs.append(r);pending['index']+=1
            if pending['index']==pending['beats']:pending=None
    if not early_reset:need(pending is None,'RoPE AXI drain')
    if starts:
        need(len(rs)==164 and starts[0]['cycle']>rs[-1]['cycle'],'Actual HLS start after full input load')
        need(len(writes)<2560 if mode==91 else len(writes)==2560,'Protocol model output extent')
        for i,r in enumerate(writes):
            need(r['index']==i and r['cycle']>starts[0]['cycle'],'Protocol write order')
            healthy=not inject or mode not in (81,82,84) or r['cycle']<=inject['cycle']
            need(r['checked']==int(healthy),'Independently classify canceled writes')
            if healthy:need(r['data']==memory[i],'Identity HLS uses loaded input, not an oracle')
        if ends:
            need(ends[0]['reads']==ends[0]['writes']==2560 and ends[0]['cycle']>writes[-1]['cycle'],'Drain full protocol core')
    else:need(not writes and not ends,'No arithmetic work without start')
    if done:
        rope_status=status(mode) if 80<=mode<=88 and mode!=83 else 0
        need(done['status']==rope_status and done['hls']==0,'Upstream fault provenance')
        need(done['beats']==len(rs)-int(mode==85),'Upstream accepted load extent')
        if ends:need(done['cycle']>ends[0]['cycle'],'Core finishes before held result')
        need(handoff['cycle']>done['cycle'],'Retained upstream response before capture command')
        need({k:handoff[k] for k in ('job','sequence','epoch','layer','position','lease')}=={k:admit[k] for k in ('job','sequence','epoch','layer','position','lease')},'Full descriptor retained through RoPE')
        for r in by['CAPTURE_AR']:need(r['cycle']>handoff['cycle'],'V read after RoPE and capture admission')
    returned={r['index']:r for r in writes if r['checked']}
    for r in by['CAPTURE_QK_RETURN']:
        need(r['index'] in returned and r['data']==returned[r['index']]['data'] and r['cycle']>returned[r['index']]['cycle'],'Captured Q/K is actual owned output')
    if inject:
        need(inject['mode']==mode and inject['r']==sum(r['cycle']<=inject['cycle'] for r in rs) and inject['writes']==sum(r['cycle']<=inject['cycle'] for r in writes),'Fault timing from accepted events')
        if mode==80:need(inject['r']==8 and not starts,'Cancel during actual input AXI')
        if mode in (81,84):need(inject['reads']==64,'Fault during core read')
        if mode==82:need(inject['writes']==64,'Cancel during core write')
        if mode==83:need(done['status']==0 and not by['CAPTURE_AR'] and done['cycle']>inject['cycle'] and handoff['cycle']>inject['cycle'],'Cancel pulse at RoPE/capture boundary')
    if mode in (85,86,87,88):need(not starts and not by['CAPTURE_AR'],'Rejected input prevents downstream work')
    if mode==90:need(len(rs)==8 and not starts,'Reset during actual upstream AXI')
    if mode==91:need(len(writes)==64 and not ends,'Coordinated reset during core output')
    metrics.update(rope_beats=len(rs),protocol_writes=len(writes),protocol_starts=len(starts))
    return metrics
