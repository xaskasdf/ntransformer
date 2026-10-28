"""Bounded independent witness for native NTRPUF01 mapping, not production assembly."""
import re
from functools import lru_cache
from check_ntrpu_pcie_bypass_native import bits,require,sample_vcd
from check_ntrpu_pcie_bypass_smoke import fields,packet_blocks

ROOT='/board/ntrpu_bypass/'
EXTRA={ROOT+k:w for k,w in dict(aw_pending=1,w_pending=1,lease_window=64,lease_epoch=64,lease_tag=64,
    decode_accept=1,decode_faults=7,decoded_bitmap=64,decoded_data=512,staged=512,staged_valid=64,current_job=32).items()}
EXTRA['/board/EP/m_axib_awlock']=1
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
    require(len(names)==61 and set(names)==set(native['widths'])|set(EXTRA),'exact61 signal inventory')
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
        require(bits(b['id'])==0 and bits(b['resp'])==(0 if accepted else 2),'AXI B verdict')
        require(int(hw['ps'])<=first['ps']<last['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<d['ps']<b['ps'],'write/decode/response order')
        for c,e in [('AW',aw),('W',w),('B',b)]:
            l=rows[c][n];require(int(l['ps'])==e['ps'],'AXI log timestamp '+c)
            for k,value in e.items():
                if k in ['channel','ps']:continue
                require((hexraw(l[k],512)==value if k=='data' else int(l[k],10 if k in ['len','size','burst','last'] else 16)==bits(value)),'AXI logged value '+c+k)
            if c=='AW':require(l['lock']=='0','logged AWLOCK')
        measurements.append(dict(n=n,job=j,fault=fault,first_rq_ps=first['ps'],last_rq_ps=last['ps'],decode_ps=d['ps'],b_ps=b['ps']))
    require(commits==[dict(job=int(l['job']),data=int(l['data'],16),ps=int(l['ps'])) for l in rows['COMMIT']] and [c['job'] for c in commits]==[0,1,2,3],'exact BFM line commits')
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
    return dict(status='PASS_NATIVE_FRAGMENT64_MAPPING',qualified=True,writes=16,accepted=9,rejected=7,bfm_lines=4,reads=12,native_handshakes={k:len(v) for k,v in groups.items()},decisions=measurements,native_stalls=native['stalls'],end_ps=finish,
        scope='Native two-beat512-bit PCIe producer to one64-byte XDMA AXI transfer and unchanged production decoder. Lease/assembly/memory are BFMs. No DDR, production capture/assembly, upstream fence, reset, physical timing or throughput qualification.')
