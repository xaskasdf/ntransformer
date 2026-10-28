"""Independent native NTRPUF01 fixture and complete RQ/CDC frame witness."""
from check_ntrpu_pcie_bypass_native import bits,require
from check_ntrpu_pcie_bypass_smoke import fields,packet_blocks

BASES=[0,0x80000,0x80000,0xfffc0]


def fixture():
    result=[]
    for job in range(4):
        for offset,kind in [(0,job+1),(0,0)]+([(0,0)] if job==1 else [])+([(32,0)] if job!=2 else []):
            magic=0x313046555052544e^(kind==1)
            epoch=0x100000001+job;tag=0x8000000100002000+job;payload_job=job
            if kind==2:tag^=1<<40
            if kind==3:epoch=0x100000002;tag=0x8000000100002001;payload_job=1
            frame=b''.join(v.to_bytes(8,'little') for v in [magic,epoch,tag,offset])
            frame+=bytes((payload_job*53+(offset+i)*29+23)&255 for i in range(32))
            fault={0:0,1:8,2:32,3:48,4:4}[kind]
            result.append(dict(job=job,offset=offset,kind=kind,frame=frame,base=0 if kind==4 else BASES[job],
                fault=fault,resp=0 if kind==0 else 3 if kind==4 else 2,code={0:0,1:2,2:7,3:7,4:3}[kind]))
    require(len(result)==12,'independent frame fixture size')
    return result


def split_requests(requests):
    """Consume every RQ beat once; continuation payload is never an address."""
    data=[];control=[];i=0
    while i<len(requests):
        first=requests[i];address=bits(first['tdata'],63,0)
        if (9<<32)<=address<(9<<32)+(1<<20):
            require(i+1<len(requests),'truncated frame')
            last=requests[i+1]
            require(bits(first['tlast'])==0 and bits(last['tlast'])==1,'bounded two-beat packet')
            data.append((first,last));i+=2
        else:
            require(address<(1<<16) and bits(first['tlast'])==1,'single-beat control BAR0 packet')
            control.append(first);i+=1
    require(len(data)==12 and len(control)==154 and len(requests)==178,'exact RQ partition')
    return data,control


def review_frames(sim,tx_text,native,aux,rows,start,finish):
    requests=[e for e in native['events'] if e['channel']=='RQ' and e['ps']>=start]
    packets,_=split_requests(requests)
    groups={c:[e for e in native['events'] if e['channel']==c and e['ps']>=start] for c in ['AW','W','B','AR','R']}
    groups['RQ']=[first for first,last in packets]
    require(all(len(groups[c])==12 for c in ['RQ','AW','W','B']),'native frame counts')
    require(not groups['AR'] and not groups['R'],'no bypass readback')
    device={c:[e for e in aux['events'] if e['channel']==c] for c in ['DAW','DW','DB']}
    require(all(len(v)==12 for v in device.values()),'CDC frame conservation')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start and (9<<32)<=((p['values'].get('Address High',0)<<32)|p['values'].get('Address Low',0))<(9<<32)+(1<<20)]
    host=[fields(l) for l in sim.splitlines() if l.startswith('NTRPU_NATIVE_FRAME_WRITE ')]
    require(len(tx)==len(host)==12 and all(len(rows(c))==12 for c in ['AW','W','B']),'frame logs count')
    fixtures=fixture();banks={j:[None]*64 for j in range(4)};bitmaps={j:0 for j in range(4)}
    measurements=[];accepted_bytes=0;rejected=identity=0;last_code=last_fault=0
    telemetry_steps=[]
    for i,f in enumerate(fixtures):
        job=f['job'];frame=f['frame'];base=f['base'];first,last=packets[i]
        aw,w,b=[groups[c][i] for c in ['AW','W','B']];h=host[i]
        require(set(h)=={'n','job','addr','data','kind','tag','ps'},'frame intent schema')
        require(int(h['n'])==i+1 and int(h['job'])==job and int(h['kind'])==f['kind'],'frame intent sequence')
        require(int(h['addr'],16)==(9<<32)+base and int(h['data'],16)==int.from_bytes(frame,'little') and int(h['tag'],16)==0x40+i,'frame intent bytes/address/tag')
        for req,keep,eop in [(first,65535,0),(last,15,1)]:
            require(bits(req['tkeep'])==keep and bits(req['tlast'])==eop,'two-beat RQ keep/last')
        d=first['tdata']
        require(bits(d,63,0)==(9<<32)+base and bits(d,74,64)==16 and bits(d,78,75)==1,'RQ write address/type/count')
        require(bits(d,79,79)==0 and bits(d,127,120)==0,'RQ frame flags')
        require(bits(d,95,80)==0xaf and bits(d,119,104)==0x1a0 and bits(d,103,96)==0x40+i,'RQ frame IDs/tag')
        require(bits(first['tuser'],15,0)==0xf0f and bits(first['tuser'],25,16)==16 and bits(first['tuser'],35,26)==0,'RQ SOP/byte enables')
        require(bits(last['tuser'],15,0)==0xf0f and bits(last['tuser'],25,16)==0 and bits(last['tuser'],35,26)==13,'RQ EOP/byte enables')
        payload=bits(d,511,128).to_bytes(48,'little')+bits(last['tdata'],127,0).to_bytes(16,'little')
        require(payload==frame and bits(last['tdata'],511,128)==0,'all 64 native frame bytes')
        p=tx[i];v=p['values']
        require(p['kind']=='Memory Write-64 Frame' and p['ps']==last['ps'] and p['payload']==frame,'decoded full frame payload/time')
        require((v['Address High']<<32)|v['Address Low']==(9<<32)+base and v['Tag']==0x40+i and v['Length']==16 and v['Last and First Byte Enables']==255,'decoded frame header')
        require(v['Requester Id']==0xaf,'decoded frame requester')
        require(bits(aw['addr'])==(1<<32)+base and bits(aw['id'])==bits(aw['len'])==0 and bits(aw['size'])==6 and bits(aw['burst'])==1,'translated 64-byte AW')
        require(bits(w['strb'])==(1<<64)-1 and bits(w['last'])==1 and bits(w['data'])==int.from_bytes(frame,'little'),'entire native AXI frame')
        require(bits(b['id'])==0 and bits(b['resp'])==f['resp'],'native B frame verdict')
        da,dw,db=[device[c][i] for c in ['DAW','DW','DB']]
        require(all(da[k]==bits(aw[k]) for k in ['addr','id','len','size','burst']) and da['lock']==0,'CDC AW payload')
        require(dw['strb']==bits(w['strb']) and dw['last']==1 and bits(dw['data'])==bits(w['data']),'all CDC frame bytes')
        require(db['id']==bits(b['id']) and db['resp']==bits(b['resp']),'CDC B payload')
        require(start<=int(h['ps'])<=first['ps']<last['ps']<=min(aw['ps'],w['ps'])<b['ps']<finish,'native frame causal order')
        require(da['ps']>aw['ps'] and dw['ps']>w['ps'] and db['ps']>max(da['ps'],dw['ps']) and b['ps']>db['ps'],'CDC causal order')
        if f['fault']:
            rejected+=1;identity+=int(f['fault'] in [32,48]);last_code=f['code'];last_fault=f['fault']
        else:
            for lane,value in enumerate(frame[32:],f['offset']):
                require(banks[job][lane] is None or banks[job][lane]==value,'identical replay only')
                banks[job][lane]=value;bitmaps[job]|=1<<lane
            accepted_bytes+=32
        state=aux['states'][db['ps']]
        require(state['epoch']==0x100000001+job and state['valid_bytes']==bitmaps[job],'staging identity/bitmap')
        require(state['failed']==state['error_code']==0,'untrusted frame must not poison lease')
        require((state['rejected_frames'],state['rejected_identity'],state['last_reject_code'],state['last_reject_faults'])==(rejected,identity,last_code,last_fault),'actual rejection telemetry')
        if f['fault']:
            telemetry_steps.append((max(da['ps'],dw['ps']),db['ps'],(rejected,identity,last_code,last_fault)))
        for c,e in [('AW',aw),('W',w),('B',b)]:
            logged=rows(c)[i]
            require(set(logged)==set(e)-{'channel'} and int(logged['ps'])==e['ps'],'native AXI log schema/time')
            for k,val in e.items():
                if k not in ['channel','ps']:
                    require(int(logged[k],10 if k in ['len','size','burst','last','resp'] else 16)==bits(val),'native AXI log '+c+k)
        measurements.append(dict(job=job,first_rq_ps=first['ps'],last_rq_ps=last['ps'],b_ps=b['ps'],rq_to_b_ps=b['ps']-first['ps'],resp=f['resp'],fault=f['fault']))
    require(accepted_bytes==256 and rejected==4 and identity==2,'frame accounting')
    # Counters persist across leases. Each change must correspond to exactly
    # one rejected frame, after its AW/W and no later than its UI B response.
    previous=(0,0,0,0);cursor=0
    for ps,state in sorted(aux['states'].items()):
        value=tuple(state[k] for k in ['rejected_frames','rejected_identity','last_reject_code','last_reject_faults'])
        if value!=previous:
            require(cursor<len(telemetry_steps),'unexpected telemetry change')
            begin,end,expected=telemetry_steps[cursor]
            require(begin<ps<=end and value==expected,'telemetry change identity/order')
            cursor+=1;previous=value
        if state['epoch']!=0x100000003:
            require(state['failed']==state['error_code']==0,'untrusted frames transiently poisoned lease')
    require(cursor==4 and previous==(4,2,3,4),'final telemetry retention')
    return groups,device,fixtures,banks,bitmaps,measurements,accepted_bytes
