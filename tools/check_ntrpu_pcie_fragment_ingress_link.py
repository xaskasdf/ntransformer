"""Native XDMA and production NTRPUF01 collector witness; control/destination remain BFMs."""
import re
from functools import lru_cache
from check_ntrpu_pcie_bypass_native import bits,require,sample_vcd
from check_ntrpu_pcie_bypass_smoke import fields,packet_blocks

ROOT='/board/ntrpu_bypass/'
EXTRA={ROOT+k:w for k,w in dict(aw_pending=1,w_pending=1,lease_window=64,lease_epoch=64,lease_tag=64,
    decode_accept=1,decode_faults=7,decoded_bitmap=64,decoded_data=512,staged=512,staged_valid=64,current_job=32).items()}
EXTRA['/board/EP/m_axib_awlock']=1
LIFECYCLE=dict(stop=1,source_stopped=1,ingress_isolated=1,quiesce_epoch=64,
    arm_valid=1,arm_ready=1,arm_epoch=64,arm_tag=64,arm_window=64,arm_destination=64,
    epoch=64,lease_held=1,failed=1,error_code=4,transport_idle=1,rejected_frames=32,rejected_identity=32,last_reject_code=4,last_reject_faults=7,
    line_valid=1,line_ready=1,line_epoch=64,line_tag=64,line_address=64,line_data=512,
    done_valid=1,done_ready=1,done_epoch=64,done_tag=64,
    result_valid=1,result_ready=1,result_epoch=64,result_tag=64,result_status=4)
EXTRA.update({ROOT+k:w for k,w in LIFECYCLE.items()})
BASES=[0,0x80000,0xfffc0,0x80000]
FAULTS=[0]*6+[0x30,8,0x20,0x40,2,0x10,0,0,0,4]


def line_bytes(job):return bytes((job*53+i*29+23)&255 for i in range(64))


def fixture(n):
    job=n//2 if n<6 else 3
    offset=(n%2)*32 if n<6 else (32 if n in [12,13] else 0)
    if n==9:offset=64
    epoch=0x100000001+job;tag=0x8000000100200300+job
    if n in [6,11]:epoch=0x100000002
    if n==6:tag=0x8000000100200301
    if n==8:tag^=1<<40
    magic=0x313046555052544e^(n==7)
    payload=bytes(((1 if n==6 else job)*53+(offset+i)*29+23)&255 for i in range(32))
    frame=b''.join(x.to_bytes(8,'little') for x in [magic,epoch,tag,offset])+payload
    return dict(job=job,offset=offset,frame=frame,base=0xfffc0 if n==15 else BASES[job],fault=FAULTS[n],last=7 if n==10 else 15)


@lru_cache(maxsize=1)
def sample(text):
    header,body=text.split('$enddefinitions $end',1)
    names={};widths={};scope=[];filtered_header=[]
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+ '/'.join(scope+[f[4]])
            require(name not in names,'duplicate VCD variable')
            names[name]=f[3];widths[f[3]]=int(f[2])
            if name in EXTRA:require(int(f[2])==EXTRA[name],'extra width '+name);keep=False
        if keep:filtered_header.append(line)
    require(set(EXTRA).issubset(names),'missing fragment signals')
    codes={c for n,c in names.items() if n not in EXTRA}
    def decode(line):return line[1:].split() if line.startswith('b') else (line[0],line[1:])
    filtered=[l for l in body.splitlines() if not l.strip() or l.strip()[0] in '$#' or decode(l.strip())[1] in codes]
    native=sample_vcd('\n'.join(filtered_header)+'\n$enddefinitions $end\n'+'\n'.join(filtered))
    require(len(names)==95 and set(names)==set(native['widths'])|set(EXTRA),'exact95 signal inventory')
    values={};updates={};now=0;states=[];rq_open=False
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        nonlocal rq_open
        if values.get(names['/board/RP/user_clk'])=='0' and updates.get(names['/board/RP/user_clk'])=='1':
            valid=raw('/board/RP/s_axis_rq_tvalid');ready=raw('/board/RP/s_axis_rq_tready')
            if rq_open:require(valid=='1','RQ VALID dropped within packet')
            if valid=='1' and ready=='1':rq_open=not bool(bits(raw('/board/RP/s_axis_rq_tlast')))
        if values.get(names['/board/EP/user_clk'])=='0' and updates.get(names['/board/EP/user_clk'])=='1' and raw('/board/EP/user_resetn')=='1':
            if raw('/board/EP/m_axib_awvalid')=='1':require(raw('/board/EP/m_axib_awlock')=='0','native AWLOCK')
            state={n.removeprefix(ROOT):raw(n) for n in EXTRA if n.startswith(ROOT)}
            state['ps']=now;states.append(state)
        values.update(updates)
    for l in body.splitlines():
        l=l.strip()
        if not l or l[0]=='$':continue
        if l[0]=='#':flush();updates={};now=int(l[1:]);continue
        raw_value,code=decode(l)
        require(code in widths and set(raw_value.lower())<=set('01xz'),'invalid extra VCD')
        raw_value=raw_value.lower();require(len(raw_value)<=widths[code],'overwide extra VCD')
        updates[code]=raw_value.rjust(widths[code],raw_value[0] if raw_value[0] in 'xz' else '0')
    flush();require(not rq_open,'unfinished RQ packet')
    return native,states


def hexraw(value,width):
    table={**{f'{i:x}':f'{i:04b}' for i in range(16)},'x':'xxxx','z':'zzzz'}
    require(len(value)*4==width,'logged raw width')
    return ''.join(table[c] for c in value.lower())


def review_lifecycle(sim,states,rows,decisions,groups,start):
    events=[(l.split()[0].removeprefix('NTRPU_COLLECTOR_'),fields(l)) for l in sim.splitlines() if l.startswith('NTRPU_COLLECTOR_')]
    expected_counts=dict(ARM=4,LINE=4,DONE=4,RESULT=4,PASS=1)
    require(set(k for k,_ in events)==set(expected_counts),'exact collector markers')
    logged={k:[v for n,v in events if n==k] for k in expected_counts}
    require(all(len(logged[k])==v for k,v in expected_counts.items()),'collector marker counts')
    sampled={k:[] for k in ['ARM','LINE','DONE','RESULT']};held={};stalls={k:0 for k in sampled}
    selectors=dict(ARM=('arm_', ['epoch','tag','window','destination']),LINE=('line_', ['epoch','tag','address','data']),
                   DONE=('done_', ['epoch','tag']),RESULT=('result_', ['epoch','tag','status']))
    rejected=identity=0;decision_index=0
    for state in states:
        if state['ps']<start:continue
        s={k:bits(state[k]) for k in LIFECYCLE};s['ps']=state['ps']
        require(s['stop']==s['failed']==s['error_code']==0,'native collector no lease failure')
        require(s['rejected_frames']==rejected and s['rejected_identity']==identity,'cycle-exact rejection telemetry')
        if state['aw_pending']==state['w_pending']=='1':
            require(decision_index<16 and decisions[decision_index]['ps']==s['ps'],'paired-decision order')
            faults=FAULTS[decision_index]
            if faults:rejected+=1
            if faults and not faults&~0x30:identity+=1
            decision_index+=1
        for name,(prefix,fs) in selectors.items():
            payload={k:s[prefix+k] for k in fs};valid=s[prefix+'valid'];ready=s[prefix+'ready']
            if name in held:require(valid==1 and payload==held.pop(name),'stable collector '+name)
            if valid:
                if ready:
                    sampled[name].append(dict(ps=s['ps'],**payload))
                    if name in ['ARM','RESULT']:
                        require(s['source_stopped']==s['ingress_isolated']==s['transport_idle']==1 and s['quiesce_epoch']==s['epoch'],'trusted barrier and transport idle at '+name)
                else:held[name]=payload;stalls[name]+=1
    require(not held and decision_index==16 and rejected==7 and identity==3,'collector final counters/drain')
    require(all(len(v)==4 for v in sampled.values()),'four native collector lifecycles')
    publication_indices=[1,3,5,14]
    for j in range(4):
        arm,line,done,result=[sampled[k][j] for k in ['ARM','LINE','DONE','RESULT']]
        epoch=0x100000001+j;tag=0x8000000100200300+j;address=(1<<32)+BASES[j]
        require(arm['epoch']==line['epoch']==done['epoch']==result['epoch']==epoch and arm['tag']==line['tag']==done['tag']==result['tag']==tag,'full64 lifecycle identity')
        require(arm['window']==arm['destination']==line['address']==address and line['data']==int.from_bytes(line_bytes(j),'little'),'authorized production line')
        require(arm['ps']<groups['RQ'][2*(j*2 if j<3 else 6)]['ps'],'ARM before frame requests')
        require(groups['B'][publication_indices[j]]['ps']<line['ps']<done['ps']<result['ps'] and result['status']==0,'publication/B/destination/result order')
        if j:require(sampled['RESULT'][j-1]['ps']<arm['ps'],'retirement before next ARM')
        commit=rows['COMMIT'][j];lease=rows['LEASE'][j]
        require(int(commit['ps'])==line['ps'] and int(lease['ps'])>arm['ps'],'real commit/lease markers')
        for k,e in [('ARM',arm),('LINE',line),('DONE',done),('RESULT',result)]:
            l=logged[k][j];require(int(l['ps'])==e['ps'],'logged lifecycle time')
            for field,value in e.items():
                if field=='ps':continue
                label='dest' if field=='destination' else 'addr' if field=='address' else field
                require(int(l[label],10 if field=='status' else 16)==value,'logged lifecycle '+k+field)
            if k=='DONE':require(l['error']=='0','BFM destination error flag')
    final=logged['PASS'][0]
    require([int(final[k]) for k in ['arms','lines','done','results','rejected','identity']]==[4,4,4,4,7,3],'collector final marker')
    require(int(final['line_stall'])==stalls['LINE'] and int(final['ps'])==int(rows['PASS'][0]['ps']),'collector observed stalls/end')
    require(stalls['LINE']>0 and stalls['RESULT']>0,'native line/result backpressure coverage')
    return dict(sampled=sampled,stalls=stalls,rejected_frames=rejected,rejected_identity=identity)


def review(sim,tx_text,rx_text,vcd):
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',sim,re.M|re.I),'simulation diagnostic')
    events=[(l.split()[0].removeprefix('NTRPU_FRAME_'),fields(l)) for l in sim.splitlines() if l.startswith('NTRPU_FRAME_')]
    counts=dict(ENUM=1,LEASE=4,WRITE=16,AW=16,W=16,B=16,DECODE=16,COMMIT=4,READ=12,VALUE=12,AR=12,R=12,COUNTS=1,PASS=1)
    require(set(n for n,_ in events)==set(counts),'exact fixture marker inventory')
    rows={n:[v for k,v in events if k==n] for n in counts}
    require(all(len(rows[n])==c for n,c in counts.items()),'fixture marker counts')
    enum=rows['ENUM'][0];final=rows['PASS'][0];summary=rows['COUNTS'][0]
    require([enum['type'+str(i)] for i in range(3)]==['10','10','11'],'BAR types')
    require(int(enum['bar0'],16)==0 and int(enum['bar1'],16)==0x10000 and int(enum['bar2'],16)==0 and int(enum['bar3'],16)==9 and enum['dma_bar']=='1','BAR assignment')
    require(int(enum['range2'],16)&0xfffffff0==0xfff00000,'1MiB aperture')
    require([int(final[k]) for k in ['writes','reads','accepted','rejected','assembled_lines']]==[16,12,9,7,4],'final counters')
    require([int(summary[k]) for k in ['aw','w','b','ar','r','decoded','accepted','rejected','committed']]==[16,16,16,12,12,16,9,7,4],'AXI counters')
    native,states=sample(vcd);start=int(enum['ps']);finish=int(final['ps'])
    require(native['end_ps']==finish==int(summary['ps']),'final VCD drain')
    require(all(int(summary[k+'_stall'])==native['stalls'][k.upper()] for k in ['aw','w','ar']),'native stalls')
    groups={c:[e for e in native['events'] if e['ps']>=start and e['channel']==c] for c in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require({c:len(e) for c,e in groups.items()}==dict(RQ=44,RC=12,TAG=12,AW=16,W=16,B=16,AR=12,R=12),'native handshake counts')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start];rx=[p for p in packet_blocks(rx_text) if p['ps']>=start]
    require(len(tx)==28 and len(rx)==12,'decoded TLP counts')
    decisions=[s for s in states if s['ps']>=start and s['aw_pending']==s['w_pending']=='1']
    require(len(decisions)==16,'actual paired decoder decisions')
    staging=bytearray(64);bitmap=0;job=-1;planned={};commits=[];measurements=[]
    for n in range(16):
        f=fixture(n);j=f['job'];base=f['base'];frame=f['frame'];fault=f['fault'];mask=((1<<64)-1)>>(n==10)
        hw=rows['WRITE'][n];aw,w,b=[groups[c][n] for c in ['AW','W','B']];d=decisions[n]
        first,last=groups['RQ'][2*n:2*n+2]
        require(int(hw['n'])==n and int(hw['job'])==j and int(hw['addr'],16)==(9<<32)+base,'host write identity')
        require(int(hw['data'],16)==int.from_bytes(frame,'little') and hw['first']=='f' and int(hw['last'],16)==f['last'] and int(hw['tag'],16)==0x40+n,'host write bytes/masks/tag')
        for req,keep,eop in [(first,65535,0),(last,15,1)]:
            require(bits(req['tkeep'])==keep and bits(req['tlast'])==eop,'two-beat RQ keep/last')
        require(bits(first['tdata'],63,0)==(9<<32)+base and bits(first['tdata'],74,64)==16 and bits(first['tdata'],78,75)==1,'RQ write address/type/count')
        require(bits(first['tdata'],79,79)==0 and bits(first['tdata'],127,120)==0,'RQ write flags')
        require(bits(first['tdata'],95,80)==0xaf and bits(first['tdata'],119,104)==0x1a0 and bits(first['tdata'],103,96)==0x40+n,'RQ write IDs/tag')
        require(bits(first['tuser'],7,0)==15 and bits(first['tuser'],15,8)==f['last'] and bits(first['tuser'],25,16)==16 and bits(first['tuser'],35,26)==0,'first RQ sideband')
        require(bits(last['tuser'],25,16)==0 and bits(last['tuser'],35,26)==13,'last RQ sideband')
        payload=bits(first['tdata'],511,128).to_bytes(48,'little')+bits(last['tdata'],127,0).to_bytes(16,'little')
        require(payload==frame and bits(last['tdata'],511,128)==0,'native all64 RQ bytes')
        p=tx[n];v=p['values']
        require(p['kind']=='Memory Write-64 Frame' and p['ps']==last['ps'] and p['payload']==frame,'serialized write payload/time')
        require((v['Address High']<<32)|v['Address Low']==(9<<32)+base and v['Tag']==0x40+n and v['Length']==16 and v['Last and First Byte Enables']==(f['last']<<4)|15,'serialized write header')
        require(bits(aw['addr'])==(1<<32)+base and bits(aw['id'])==0 and bits(aw['len'])==0 and bits(aw['size'])==6 and bits(aw['burst'])==1,'AXI full64-byte profile')
        require(bits(w['strb'])==mask and bits(w['last'])==1,'full orpartial AXI mask')
        for i,byte in enumerate(frame):
            if mask>>i&1:require(bits(w['data'],i*8+7,i*8)==byte,'active AXI frame byte')
        if job!=j:job=j;staging=bytearray(64);bitmap=0
        require(bits(d['current_job'])==j and bits(d['lease_window'])==(1<<32)+BASES[j] and bits(d['lease_epoch'])==0x100000001+j and bits(d['lease_tag'])==0x8000000100200300+j,'actual full64 lease identity')
        require(bits(d['staged'])==int.from_bytes(staging,'little') and bits(d['staged_valid'])==bitmap,'predecision staged state')
        accepted=fault==0;dm=(0xffffffff<<f['offset']) if accepted else 0
        require(bits(d['decode_accept'])==accepted and bits(d['decode_faults'])==fault and bits(d['decoded_bitmap'])==dm,'native decoder decision')
        logged=rows['DECODE'][n]
        require(int(logged['n'])==n and int(logged['ps'])==d['ps'] and int(logged['accept'])==accepted and int(logged['faults'],16)==fault and int(logged['mask'],16)==dm and hexraw(logged['data'],512)==d['decoded_data'],'logged decoder decision')
        if accepted:
            decoded=int.from_bytes(frame[32:],'little')<<(f['offset']*8)
            require(bits(d['decoded_data'])==decoded,'decoded positioned payload')
            old=bitmap;bitmap|=dm;staging[f['offset']:f['offset']+32]=frame[32:]
            if bitmap==(1<<64)-1 and old!=bitmap:commits.append(dict(job=j,data=int.from_bytes(staging,'little'),ps=d['ps']))
        planned[d['ps']]=(j,bytes(staging),bitmap)
        require(bits(b['id'])==0 and bits(b['resp'])==(0 if accepted else 3 if n==15 else 2),'AXI B verdict')
        require(int(hw['ps'])<=first['ps']<last['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<d['ps']<b['ps'],'write/decode/response order')
        for c,e in [('AW',aw),('W',w),('B',b)]:
            l=rows[c][n];require(int(l['ps'])==e['ps'],'AXI log timestamp '+c)
            for k,value in e.items():
                if k in ['channel','ps']:continue
                require((hexraw(l[k],512)==value if k=='data' else int(l[k],10 if k in ['len','size','burst','last'] else 16)==bits(value)),'AXI logged value '+c+k)
            if c=='AW':require(l['lock']=='0','logged AWLOCK')
        measurements.append(dict(n=n,job=j,fault=fault,first_rq_ps=first['ps'],last_rq_ps=last['ps'],decode_ps=d['ps'],b_ps=b['ps']))
    require([c['job'] for c in commits]==[0,1,2,3],'four completed production bitmaps')
    for expected_commit,logged in zip(commits,rows['COMMIT']):
        require(int(logged['job'])==expected_commit['job'] and int(logged['data'],16)==expected_commit['data'] and int(logged['ps'])>expected_commit['ps'],'production publication after assembly')
    # Follow every sampled cycle, so a rejected fragment cannot transiently
    # damage BFM staging between two decoder observations.
    job=-1;expected=bytes(64);bitmap=0;prior=None
    for s in states:
        if s['ps']<start:continue
        j=bits(s['current_job'])
        if j==0xffffffff:continue
        if j!=job:job=j;expected=bytes(64);bitmap=0
        require(bits(s['staged'])==int.from_bytes(expected,'little') and bits(s['staged_valid'])==bitmap,'cycle-exact BFM staging preservation')
        require(bits(s['lease_window'])==(1<<32)+BASES[j] and bits(s['lease_epoch'])==0x100000001+j and bits(s['lease_tag'])==0x8000000100200300+j,'stable context for each lease')
        if s['ps'] in planned:_,expected,bitmap=planned[s['ps']]
    for j,l in enumerate(rows['LEASE']):
        require(int(l['job'])==j and int(l['window'],16)==(1<<32)+BASES[j] and int(l['epoch'],16)==0x100000001+j and int(l['tag'],16)==0x8000000100200300+j,'lease markers')
    for n in range(12):
        j=[0,3,2][n//4];lane=[0,28,32,60][n%4];offset=BASES[j]+lane;line=line_bytes(j);word=int.from_bytes(line[lane:lane+4],'little')
        req=groups['RQ'][32+n];ar,r,rc,tag=[groups[c][n] for c in ['AR','R','RC','TAG']];hr,hv=rows['READ'][n],rows['VALUE'][n]
        require(int(hr['n'])==int(hv['n'])==n and int(hr['addr'],16)==(9<<32)+offset and int(hr['expected'],16)==int(hv['data'],16)==word and int(hr['tag'],16)==0x60+n,'host readback')
        require(bits(req['tdata'],63,0)==(9<<32)+offset and bits(req['tdata'],74,64)==1 and bits(req['tdata'],79,75)==0 and bits(req['tdata'],127,120)==0,'native read descriptor')
        require(bits(req['tdata'],95,80)==0xaf and bits(req['tdata'],119,104)==0x1a0 and bits(req['tdata'],103,96)==0x60+n,'native read IDs/tag')
        require(bits(req['tkeep'])==15 and bits(req['tlast'])==1 and bits(req['tuser'],15,0)==15,'native read mask/last')
        require(bits(ar['addr'])==(1<<32)+offset and bits(ar['id'])==0 and bits(ar['len'])==0 and bits(ar['size'])==2 and bits(ar['burst'])==1,'AXI read profile')
        require(bits(r['data'])==int.from_bytes(line,'little') and bits(r['id'])==bits(r['resp'])==0 and bits(r['last'])==1,'entire assembled BFM line')
        for c,e in [('AR',ar),('R',r)]:
            l=rows[c][n];require(int(l['ps'])==e['ps'],'read log timestamp')
            for k,v in e.items():
                if k not in ['channel','ps']:require(int(l[k],10 if k in ['len','size','burst','last'] else 16)==bits(v),'read logged value')
        d=rc['tdata']
        require(bits(d,11,0)==offset%4096 and bits(d,15,12)==0 and bits(d,28,16)==4 and bits(d,30,29)==2,'RC address/count/completion')
        require(bits(d,42,32)==1 and bits(d,46,43)==0 and bits(d,71,64)==tag['tag'] and bits(d,63,48)==0xaf and bits(d,87,72)==0x1a0 and bits(d,94,89)==0 and bits(d,127,96)==word,'RC identity/status/payload')
        require(bits(rc['tkeep'])==15 and bits(rc['tlast'])==1 and bits(rc['tuser'],63,0)==0xf000 and bits(rc['tuser'],67,64)==1 and bits(rc['tuser'],96,96)==0,'RC framing')
        p=tx[16+n];v=p['values'];require(p['kind']=='Memory Read-64 Frame' and p['ps']==req['ps'] and p['payload']==b'' and v['Length']==1 and v['Tag']==0x60+n and v['Last and First Byte Enables']==15 and (v['Address High']<<32)|v['Address Low']==(9<<32)+offset,'decoded read')
        p=rx[n];v=p['values'];require(p['kind']=='Completion With Data Frame' and p['ps']==rc['ps'] and p['payload']==word.to_bytes(4,'big') and v['Tag']==tag['tag'] and v['Length']==1 and v['Completion Status']==0,'decoded completion')
        require(groups['B'][-1]['ps']<int(hr['ps'])<=req['ps']<=ar['ps']<r['ps']<=rc['ps']<=int(hv['ps']) and req['ps']<=tag['ps']<rc['ps'],'readback causal order')
    endings=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns)\s*:(.+)$',sim,re.M)
    require(len(endings) in [1,2] and len(set(endings))==1 and int(endings[0][0])*(1000 if endings[0][1]=='ns' else 1)==finish,'real simulator finish')
    lifecycle=review_lifecycle(sim,states,rows,decisions,groups,start)
    return dict(status='PASS_NATIVE_FRAGMENT_COLLECTOR',lifecycle=lifecycle,qualified=True,writes=16,accepted=9,rejected=7,production_lines=4,reads=12,native_handshakes={k:len(v) for k,v in groups.items()},decisions=measurements,native_stalls=native['stalls'],end_ps=finish,
        scope='Native XDMA to production fragment capture/assembly. ARM/barrier/destination and readback remain BFMs. No control BAR, CDC, DDR, real upstream drain, reset, physical timing or throughput qualification.')
