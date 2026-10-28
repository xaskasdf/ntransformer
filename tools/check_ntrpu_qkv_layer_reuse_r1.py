"""Reconstruct the real capture/dispatch path from accepted external data."""
import re
from check_ntrpu_qkv_dispatch_r2 import need,word

from ntrpu_qkv_layer_plan_r1 import plan,identity
SPEC=plan()

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
 'LAYERS_BIND':'cycle job group sequence epoch capacity',
 'LAYERS_STATE':'cycle job guard engine physical_checks binds',
 'LAYERS_PASS':'seed jobs frames heads words cycles qk v acks errors resets revoked'}

def parse(text):
    need(not re.search(r'(?m)^(FATAL:|Fatal:|ERROR:|FAIL\b)',text),'Failure diagnostic')
    jobs={};summary=None;cycle=0
    for line in text.splitlines():
        if not line.startswith(('CAPTURE_','DISPATCH_','CAPDISP_','LAYERS_')):continue
        tag,*tokens=line.split();need(tag in SCHEMA,'Known event')
        fields=[t.split('=') for t in tokens]
        need(all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split(),'Exact fields')
        row={}
        for key,val in fields:
            width=72 if key in ('guard','engine') else 512 if key=='query' else 128 if key=='data' and tag in ('CAPTURE_R','DISPATCH_RAM','DISPATCH_KV','DISPATCH_Q','DISPATCH_W') else 8 if key=='data' else 16 if key in ('sequence','epoch','base','lease','va') or (key=='address' and tag in ('CAPTURE_AR','CAPTURE_R','DISPATCH_AW')) else 0
            need(re.fullmatch('[0-9a-f]{'+str(width)+'}' if width else '[0-9]+',val) is not None,'Numeric shape');row[key]=int(val,16 if width else 10)
        if 'cycle' in row:need(row['cycle']>=cycle,'Monotonic event time');cycle=row['cycle']
        need(summary is None,'No events after summary')
        if tag=='LAYERS_PASS':summary=row;continue
        if tag=='LAYERS_BIND':
            need(row['job']==len(jobs)+1,'Bind precedes new job');jobs[row['job']]=[]
        if tag=='CAPTURE_BEGIN':
            if row['job']==len(jobs)+1:jobs[row['job']]=[]
            else:need(row['job']==len(jobs) and len(jobs[row['job']])==1 and jobs[row['job']][0][0]=='LAYERS_BIND','One begin after bind')
        need(row['job']==len(jobs) and row['job'] in jobs,'Current job event');jobs[row['job']].append((tag,row))
    need(len(jobs)==58 and summary is not None,'Complete bank')
    return jobs,summary

def review_job(job,events):
    spec=SPEC[job-1];mode=spec['mode'];want=spec['want'];seq,epoch=identity(spec['group'])
    pos=spec['position'];layer=spec['layer'];cap=3;base=0x900000000fc0;va=0xc000000fc0
    lease=0xffffffffffffffe0 if mode==2 else 0x2000000000001000+job*64
    need(events[0]==('CAPTURE_BEGIN',dict(job=job,sequence=seq,epoch=epoch,position=pos,va=va)),'Capture descriptor')
    need(events[1]==('DISPATCH_BEGIN',dict(job=job,mode=mode,sequence=seq,epoch=epoch,layer=layer,position=pos,base=base,capacity=cap)),'Dispatch descriptor')
    by={tag:[] for tag in SCHEMA};last=0
    for tag,row in events[2:]:
        need(row['job']==job and row['cycle']>=last,'Ordered job events');last=row['cycle'];by[tag].append(row)
    def one(tag,required=True):
        need(len(by[tag])==int(required),'Unique/required '+tag);return by[tag][0] if required else None
    admit=one('DISPATCH_ADMIT');release=one('CAPDISP_RELEASE');reset=one('CAPTURE_RESET',mode in (50,51,52,53))
    cc=one('CAPTURE_COMPLETE',mode not in (50,51));complete=one('CAPDISP_COMPLETE',mode not in (50,51,52,53))
    ack=one('DISPATCH_ACK',mode not in (50,51,52,53));ca=one('DISPATCH_CAPTURE_ACK',mode not in (50,51,52,53));sa=one('CAPTURE_SOURCE_ACK',mode not in (50,51,52,53))
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
        cap_status=spec['capture_status']
        need(cc['status']==cap_status,'Capture preserves its own first fault')
        stored=[]
        for r in returns:
            if ((r['data']>>23)&255)==255 or (inject and mode in (24,25,26,27) and r['cycle']>inject['cycle']):break
            stored.append(r)
        need(cc['qk']==len(stored) and cc['v']==len(rs)-int(mode==20),'Actual capture write extents')
        need(cc['frames']==(192 if cap_status==0 and mode!=28 else 0),'Capture publication extent')
        if cap_status==0:need(len(returns)==2560 and len(rs)==32,'Complete source before handoff')
        if mode in (10,11,12,13,91,92):need(not rs and not requests and not by['CAPTURE_AR'],'Invalid capture command has no reads')
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
        if 10<=mode<=28 or mode>=90:need(app is None and not kv and root is None,'Capture failure prevents side effects')
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
    need((release['aw'],release['w'],release['b'],release['capture_acks'],release['rope_acks'],release['reset'])==(len(aws),len(ws),len(bs),int(mode not in (50,51,52,53)),int(mode not in (50,51,52,53)),int(mode in (50,51,52,53))),'Release counters')
    return dict(job=job,mode=mode,frames=len(kv)+len(qs),heads=len(heads),words=len(results),qk=len(requests),v=len(rs),acks=int(mode not in (50,51,52,53)),
        errors=int(want!=0),resets=int(mode in (50,51,52,53)),revoked=int(revoke is not None),admission_cycle=admit['cycle'],end_cycle=release['cycle'])

def review_parsed(jobs,summary,seed):
    need(list(jobs)==list(range(1,59)),'Job inventory')
    metrics=[];prefixes=[0]*16;writers={};memory={};checks=0;binds=0;last_release=0
    for job,events in jobs.items():
        spec=SPEC[job-1];by={t:[r for tag,r in events if tag==t] for t in SCHEMA}
        need(len(by['LAYERS_BIND'])==int(spec['rebind']),'Only planned binds')
        if spec['rebind']:
            r=by['LAYERS_BIND'][0];seq,epoch=identity(spec['group'])
            need(events[0][0]=='LAYERS_BIND' and r==dict(cycle=r['cycle'],job=job,group=spec['group'],sequence=seq,epoch=epoch,capacity=3),'Full-width rebind identity')
            need(last_release<r['cycle']<by['DISPATCH_ADMIT'][0]['cycle'],'Bind between drained commands')
            prefixes=[0]*16;writers={};binds+=1
        transport=[e for e in events if not e[0].startswith('LAYERS_')]
        m=review_job(job,transport);metrics.append(m)
        need(m['admission_cycle']>last_release,'Exclusive command lifetime');last_release=m['end_cycle']
        pending=None;burst=[]
        for tag,r in events:
            if tag=='DISPATCH_AW':pending=r
            elif tag=='DISPATCH_W':burst.append(r)
            elif tag=='DISPATCH_B':
                need(pending is not None and len(burst)==pending['beats'],'Persistent memory owns complete burst')
                if r['response']==0:
                    for i,w in enumerate(burst):memory[pending['address']+i*64]=w['data']
                pending=None;burst=[]
        need(pending is None and not burst,'Persistent memory drained')
        c=by['CAPDISP_COMPLETE'][0];layer=spec['layer'];pos=spec['position']
        if c['committed']:
            need(0<=layer<16 and pos==prefixes[layer]<3,'Only next token of its own layer may publish')
            prefixes[layer]+=1;writers[layer,pos]=job
        need(len(by['LAYERS_STATE'])==1 and events[-2][0]=='LAYERS_STATE','One final metadata observation')
        state=by['LAYERS_STATE'][0]
        expected=sum(p<<(18*i) for i,p in enumerate(prefixes))
        need(state['guard']==state['engine']==expected,'Both sixteen-layer arrays equal independent commit history')
        need(state['cycle']==m['end_cycle'] and state['binds']==binds,'Metadata observed at final drain')
        for (l,p),owner in writers.items():
            for beat in range(64):
                address=0x900000000fc0+((l*16+beat//4)*3+p)*256+(beat%4)*64
                expected=sum(word(2048+beat*16+i,owner)<<(32*i) for i in range(16))
                need(memory.get(address)==expected,'Every published historical KV frame remains intact')
                checks+=1
        need(state['physical_checks']==checks,'Cumulative published-frame coverage')
    for key in ('frames','heads','words','qk','v','acks','errors','resets','revoked'):
        need(summary[key]==sum(m[key] for m in metrics),'Summary '+key)
    need(summary['seed']==seed and summary['jobs']==58 and summary['cycles']==metrics[-1]['end_cycle'],'Summary identity')
    need(summary['words']==104448 and summary['errors']==7 and summary['resets']==0 and summary['revoked']==1,'Full multi-layer scenario')
    return dict(status='PASS_QKV_LAYER_REUSE_R1_TRACE',summary=summary,metrics=metrics,
        physical_frame_checks=checks,binds=binds,arithmetic_qualified=False)

def review(text,seed):return review_parsed(*parse(text),seed)

def controls(text,seed):
    jobs,summary=parse(text);review_parsed(jobs,summary,seed);rejected=[]
    def reject(label,j,events):
        changed=dict(jobs);changed[j]=events
        try:review_parsed(changed,summary,seed)
        except (ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corruption '+label)
    fields=[('CAPTURE_BEGIN','sequence'),('DISPATCH_BEGIN','layer'),('DISPATCH_ADMIT','lease'),
        ('CAPTURE_AR','address'),('CAPTURE_AR','beats'),('CAPTURE_R','data'),('CAPTURE_R','response'),('CAPTURE_R','last'),
        ('CAPTURE_QK','index'),('CAPTURE_QK_RETURN','data'),('CAPTURE_COMPLETE','qk'),('CAPTURE_COMPLETE','frames'),
        ('DISPATCH_RAM','data'),('DISPATCH_KV','data'),('DISPATCH_COMMIT','frames'),('DISPATCH_AW','address'),
        ('DISPATCH_W','data'),('DISPATCH_B','response'),('DISPATCH_KV_ACK','tokens'),('DISPATCH_ROOT','prefix'),
        ('DISPATCH_Q','data'),('DISPATCH_HEAD','query'),('CAPDISP_COMPLETE','committed'),('DISPATCH_RESULT','data'),
        ('CAPTURE_SOURCE_ACK','epoch'),('CAPDISP_RELEASE','rope_acks'),('LAYERS_STATE','physical_checks'),('LAYERS_STATE','binds')]
    for tag,key in fields:
        events=jobs[1].copy();i=next(i for i,(t,r) in enumerate(events) if t==tag);r=dict(events[i][1]);r[key]^=1;events[i]=(tag,r);reject(tag+'-'+key,1,events)
    for tag in ('DISPATCH_ADMIT','CAPTURE_AR','CAPTURE_R','CAPTURE_QK','CAPTURE_QK_RETURN','CAPTURE_COMPLETE','DISPATCH_APPEND','DISPATCH_KV','DISPATCH_AW','DISPATCH_W','DISPATCH_B','DISPATCH_KV_ACK','DISPATCH_ROOT','DISPATCH_Q','DISPATCH_HEAD','DISPATCH_HEAD_ACK','CAPDISP_COMPLETE','DISPATCH_RESULT','DISPATCH_ACK','DISPATCH_CAPTURE_ACK','CAPTURE_SOURCE_ACK','CAPDISP_RELEASE','LAYERS_BIND','LAYERS_STATE'):
        events=jobs[1];i=next(i for i,(t,r) in enumerate(events) if t==tag)
        reject('missing-'+tag,1,events[:i]+events[i+1:]);reject('duplicate-'+tag,1,events[:i]+[events[i]]+events[i:])
    for job in (1,19,21,51,55,58):
        for key in ('guard','engine'):
            for layer in range(16):
                events=jobs[job].copy();i=next(i for i,(t,r) in enumerate(events) if t=='LAYERS_STATE');r=dict(events[i][1]);r[key]^=1<<(18*layer);events[i]=('LAYERS_STATE',r)
                reject(f'prefix-{job}-{key}-{layer}',job,events)
    for job in (1,55):
        for key in ('sequence','epoch','group','capacity'):
            events=jobs[job].copy();i=next(i for i,(t,r) in enumerate(events) if t=='LAYERS_BIND');r=dict(events[i][1]);r[key]^=(1<<40 if key in ('sequence','epoch') else 1);events[i]=('LAYERS_BIND',r);reject(f'bind-{job}-{key}',job,events)
    for key in summary:
        bad=dict(summary);bad[key]+=1
        try:review_parsed(jobs,bad,seed)
        except ValueError:rejected.append('summary-'+key)
        else:raise AssertionError('Accepted summary '+key)
    return rejected
