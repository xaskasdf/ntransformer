"""Replay real cold-MIG writes and reads against bytes accepted by ingress.

This bounded bank covers six locations, not exhaustive capacity or the exact
DIMM. Readback is a test-only AXI producer in the UI clock domain, not a host
read protocol. Do not infer CDC, native PCIe, warm reset or inference from it.
"""
import re
from collections import Counter

LOCATIONS = [0, 0x100000040, 0x800000000, 0x1000000000, 0x1800000000, 0x1fffffffc0]
DESTINATIONS = [*LOCATIONS, 0x2000000000, 0x1000000000000000, 0xffffffffffffffc0,
                LOCATIONS[1], LOCATIONS[5], 0, 0, 0, 0]
WRITE_JOBS = {1, 2, 3, 4, 5, 6, 10, 11, 13, 15}
FIELDS = {
    'HOLD':'epoch peer idle arm result cycle',
    'TELEMETRY':'frames identity reason cycle',
    'CASE':'job name','ARM':'epoch tag window dest cycle','AW':'addr id len size cycle',
    'W':'data strb last cycle','B':'id resp cycle','MAW':'addr id len size cycle',
    'MW':'data strb last cycle','MB':'id resp cycle','PHASE':'phase busy idle status cycle',
    'RESULT':'epoch tag status cycle','STOP':'phase cycle','START':'seed ps',
    'STATE':'ps calib ui_rst rst','ACT':'ps rank count',
    'AR':'job addr cycle ps','R':'job addr data id resp last cycle ps',
    'PASS':'jobs results source_aw source_w source_b writes reads bytes cycles rank0 rank1 rank2 rank3 seed ps',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def telemetry(job):
    frames=7 if job>=14 else 6 if job>=13 else 5 if job>=12 else 4 if job>=4 else 3 if job>=3 else 2 if job>=2 else 0
    identities=3 if job>=13 else 2 if job>=3 else 1 if job>=2 else 0
    reason=1 if job>=14 else 0x307 if job>=13 else 4 if job>=12 else 0x43 if job>=4 else 0x207 if job>=3 else 0x82 if job>=2 else 0
    return frames,identities,reason


def responses(job):
    return {2:[2,2,0,0,0],3:[2,0,0],4:[3,0,0],12:[0,2],13:[2,0,0],14:[2]}.get(job,[0,0])


def packet(job,index,seed):
    address=0x1000fffc0 if job%2 else 0x100000000
    epoch=0x100000000+job;tag=0x8000000100001000+job
    offset=32*index;data_job=job
    if job==2:
        offset=[0,0,0,0,32][index]
        if index==0: epoch ^= 1<<40
    elif job in {3,4,13}:
        offset=32*max(0,index-1)
        if index==0:
            if job==3: tag ^= 1<<40
            if job==4: address ^= 1<<32
            if job==13: epoch-=2;tag-=2;data_job=11
    elif job in {12,14}: offset=0
    payload=bytes((data_job*37+i*19+seed*7)&255 for i in range(64))
    if job==12 and index==1: payload=bytes([payload[0]^1])+payload[1:]
    data=b'NTRPUF01'+epoch.to_bytes(8,'little')+tag.to_bytes(8,'little')+offset.to_bytes(8,'little')+payload[offset:offset+32]
    if job==2 and index==1: data=bytes([data[0]^1])+data[1:]
    return address,int.from_bytes(data,'little')


def review(text, seed=1):
    clean=[s for s in text.splitlines() if not s.lstrip().startswith('#')]
    diagnostic=re.compile(r'\b(?:ERROR|FATAL|FAILED|MISMATCH|VIOLATION|VIOLATED|INCORRECT_PARAMETER)\b|SIMULATIONS NOT SUPPORTED|TIMED\s+OUT|timing\s+(?:check\s+)?fail',re.I)
    need(not any(diagnostic.search(s) for s in clean),'tool/model diagnostic')
    events=[]
    for line in clean:
        if not line.startswith('NTRPU_MIG_LINE_'):continue
        label,*tokens=line.split();kind=label.removeprefix('NTRPU_MIG_LINE_')
        f=dict(v.split('=',1) for v in tokens)
        need(kind in FIELDS and len(f)==len(tokens) and set(f)==set(FIELDS[kind].split()),'trace schema')
        events.append((kind,f))
    need(events and events[-1][0]=='PASS' and sum(k=='PASS' for k,_ in events)==1,'terminal PASS')
    start=None;calibration=None;states=[];ranks=Counter();job=0
    counts=Counter();golden={};jobs=[];byte_writes=0;last_cycle=0;reading=None;read_index=0
    retired=True
    for kind,f in events:
        if 'cycle' in f:
            cycle=int(f['cycle']);need(cycle>=last_cycle,'cycle chronology');last_cycle=cycle
        if kind=='STATE':
            state={k:(v if v in {'x','z'} else int(v)) for k,v in f.items()};states.append(state)
            if calibration is None and state['calib']==1 and state['ui_rst']==0:calibration=state['ps']
            if calibration is not None and state['ps']>=calibration:
                need(state['calib']==1 and state['ui_rst']==0,'runtime calibration/reset loss')
        elif kind=='START':
            need(start is None and calibration is not None and int(f['seed'])==seed,'start identity/calibration')
            start=int(f['ps']);need(start>calibration,'admission before calibration')
        elif kind=='ACT':
            rank=int(f['rank']);need(calibration is not None and int(f['ps'])>calibration and rank in range(4),'rank activity context')
            ranks[rank]+=1;need(int(f['count'])==ranks[rank],'rank ACT count')
        elif kind=='CASE':
            need(start is not None and retired and reading is None,'unfinished prior job')
            if job:need(read_index==len(golden),'incomplete sentinel sweep')
            job+=1;need(job<=15 and int(f['job'])==job and f['name']==f'memory_job_{job}','case sequence')
            armed=retired=False;aw=w=maw=mw=mb=None;bank=[None]*64;responses_seen=0;phases=[];stops=[];holds=[];telemetry_seen=False;read_index=0
        elif kind=='ARM':
            need(job>0 and not armed and not retired,'unexpected arm')
            need(int(f['epoch'],16)==0x100000000+job and int(f['tag'],16)==0x8000000100001000+job,'arm identity')
            window=0x1000fffc0 if job%2 else 0x100000000
            need(int(f['window'],16)==window and int(f['dest'],16)==DESTINATIONS[job-1],'arm full mapping')
            armed=True;arm_cycle=cycle
        elif kind in {'AW','W'}:
            need(armed and not retired and responses_seen<len(responses(job)),'source context')
            source_address,data=packet(job,responses_seen,seed)
            if kind=='AW':
                need(aw is None and int(f['addr'],16)==source_address and int(f['len'])==0 and int(f['size'])==6,'source AW')
                need(int(f['id'],16)==(5 if job==14 else (counts['source_b']+3)%16),'source request ID')
                aw={**f,'cycle':cycle};counts['source_aw']+=1
            else:
                need(w is None and int(f['data'],16)==data and int(f['strb'],16)==(1<<64)-1 and f['last']=='1','source W')
                w={**f,'cycle':cycle};counts['source_w']+=1
        elif kind=='B':
            need(aw is not None and w is not None and cycle>max(aw['cycle'],w['cycle']),'source B before channels')
            response=responses(job)[responses_seen]
            need(int(f['resp'])==response and f['id']==aw['id'],'source response')
            if not response:
                raw=int(w['data'],16).to_bytes(64,'little');offset=int.from_bytes(raw[24:32],'little')
                need(raw[:8]==b'NTRPUF01' and int.from_bytes(raw[8:16],'little')==0x100000000+job and
                     int.from_bytes(raw[16:24],'little')==0x8000000100001000+job and offset in {0,32},'accepted fragment identity/offset')
                for lane in range(32):
                    dest=offset+lane
                    need(bank[dest] is None or bank[dest]==raw[32+lane],'conflicting accepted byte')
                    bank[dest]=raw[32+lane];byte_writes+=1
            aw=w=None;responses_seen+=1;counts['source_b']+=1;source_b=cycle
        elif kind in {'MAW','MW'}:
            need(job in WRITE_JOBS and responses_seen==len(responses(job)) and aw is w is None and None not in bank and cycle>source_b,'invalid/premature MIG write')
            if kind=='MAW':
                need(maw is None and int(f['addr'],16)==DESTINATIONS[job-1] and int(f['id'],16)==0 and int(f['len'])==0 and int(f['size'])==6,'MIG AW mapping')
                maw=cycle
            else:
                need(mw is None and int(f['data'],16).to_bytes(64,'little')==bytes(bank) and int(f['strb'],16)==(1<<64)-1 and f['last']=='1','MIG W byte reference')
                mw=cycle;counts['writes']+=1
        elif kind=='MB':
            need(maw is not None and mw is not None and mb is None and cycle>max(maw,mw),'MIG B order')
            need(int(f['id'],16)==int(f['resp'])==0,'MIG B status')
            mb=cycle;golden[DESTINATIONS[job-1]]=bytes(bank);counts['memory_b']+=1
        elif kind=='PHASE':
            phase,busy,idle,status=(int(f[k]) for k in ['phase','busy','idle','status'])
            sequence=[1,2,3,4,5,0]
            need(job not in {12,14} and len(phases)<6 and phase==sequence[len(phases)],'bridge phase sequence')
            need(busy==0 and idle==1,'phase guard/receiver idle')
            need(status==(4 if phase in {4,5,0} and job in {7,8,9} else 0),'phase destination status')
            if phase in {4,5,0} and job in WRITE_JOBS:need(mb is not None and cycle>mb,'done before memory B')
            phases.append(phase)
        elif kind=='STOP':
            need(job==10 and int(f['phase'])==3 and not stops,'stop coverage');stops.append(cycle)
        elif kind=='HOLD':
            need(job==14 and not retired and aw is not None and w is None and not phases,'queued AW hold context')
            need(int(f['epoch'],16)==int(f['peer'],16)==0x100000000+job and f['idle']==f['arm']==f['result']=='0','queued AW lease')
            need(not holds or cycle==holds[-1]+1,'hold continuity');holds.append(cycle)
        elif kind=='TELEMETRY':
            need(not retired and not telemetry_seen,'telemetry context')
            need((int(f['frames']),int(f['identity']),int(f['reason'],16))==telemetry(job),'telemetry counts/reason');telemetry_seen=True
        elif kind=='RESULT':
            need(telemetry_seen and len(holds)==(100 if job==14 else 0),'telemetry/hold coverage')
            need(armed and not retired and aw is w is None and responses_seen==len(responses(job)),'result source drain')
            status=5 if job in {7,8,9} else 1 if job in {10,14} else 4 if job==12 else 0
            need(int(f['epoch'],16)==0x100000000+job and int(f['tag'],16)==0x8000000100001000+job and int(f['status'])==status,'result identity/status')
            need(phases==([] if job in {12,14} else [1,2,3,4,5,0]),'result bridge drain')
            need(len(stops)==int(job==10),'stop count')
            if job in WRITE_JOBS:need(maw is not None and mw is not None and mb is not None and cycle>mb,'result memory drain')
            else:need(maw is mw is mb is None,'rejected write reached MIG')
            counts['results']+=1;retired=True
            jobs.append(dict(job=job,address=DESTINATIONS[job-1],arm=arm_cycle,source_b=source_b,memory_aw=maw,memory_w=mw,memory_b=mb,result=cycle,status=status))
        elif kind=='AR':
            need(retired and reading is None and read_index<len(golden) and int(f['job'])==job,'readback context')
            address=sorted(golden)[read_index]
            # Fixture traverses physical locations in ascending order.
            need(int(f['addr'],16)==address and int(f['ps'])>start,'sentinel read order')
            reading=(address,int(f['ps']),cycle);counts['ar']+=1
        elif kind=='R':
            need(reading is not None and int(f['job'])==job,'read response context')
            address,ps,ar_cycle=reading
            need(int(f['addr'],16)==address and int(f['ps'])>ps and cycle>ar_cycle,'read response order/address')
            need(int(f['data'],16).to_bytes(64,'little')==golden[address] and int(f['id'],16)==int(f['resp'])==0 and f['last']=='1','MIG readback byte reference')
            reading=None;read_index+=1;counts['reads']+=1
        elif kind=='PASS':
            summary={k:int(v) for k,v in f.items()}
            need(job==15 and retired and reading is None and read_index==6,'final sweep incomplete')
            expected=dict(jobs=15,results=15,source_aw=35,source_w=35,source_b=35,writes=10,reads=75,bytes=4800,seed=seed)
            need(all(summary[k]==v for k,v in expected.items()),'final fixture summary')
            need(counts==Counter(source_aw=35,source_w=35,source_b=35,writes=10,memory_b=10,results=15,ar=75,reads=75) and byte_writes==896,'reconstructed counts')
            need(all(ranks[i]>0 and summary[f'rank{i}']==ranks[i] for i in range(4)),'rank totals')
            need(summary['cycles']>=last_cycle and start<summary['ps']<500_000_000,'final time budget')
    endings=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns|us)\s*:(.+)$',text,re.M)
    need(len(endings) in {1,2} and len(set(endings))==1,'simulator finish')
    need(int(endings[0][0])*{'ps':1,'ns':1000,'us':1000000}[endings[0][1]]==summary['ps'],'finish time')
    return dict(status='PASS_COLD_MIG_FRAGMENT_INGRESS',qualified=True,summary=summary,calibration_ps=calibration,start_ps=start,
                accepted_byte_writes=byte_writes,memory_write_bytes=640,readback_bytes=4800,locations=LOCATIONS,jobs=jobs,
                scope='One UI clock, actual generated MIG and generic two-slot x4 ECC DDR model; external source/isolation and direct readback are test drivers. Six sparse locations, three blocked invalid destinations, current-frame conflict, stale frames after rearm, STOP with admitted AW and100cycles missing W, telemetry snapshots; not exact DIMM, warm reset, CDC, PCIe/host-read protocol, full capacity or inference qualification.')
