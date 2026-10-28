"""Native NTRD PCIe packets, observed AXI pins, CDC and two-MIG ownership."""
import re
from collections import Counter
from check_ntrpu_pcie_bypass_native import bits,require,sample_vcd
from check_ntrpu_pcie_bypass_smoke import packet_blocks
from check_ntrpu_guarded_native_transport import review as transport_review,fields,events,CTL_CHANNELS,DATA_CHANNELS
from check_ntrpu_guarded_native_core import pattern,address,periodic,rank_witness

from check_ntrpu_guarded_native_startup_r2 import EXTRA,review_startup

ROOT='/board/ntrpu_bypass/'
CONTROL_WIDTHS=dict(awaddr=32,wdata=32,araddr=32,wstrb=4,awvalid=1,wvalid=1,bready=1,arvalid=1,rready=1,
    awready=1,wready=1,arready=1,bvalid=1,rvalid=1,bresp=2,rresp=2,rdata=32)
DATA_MAP=dict(av='awvalid',ar='awready',addr='awaddr',id='awid',len='awlen',size='awsize',burst='awburst',lock='awlock',
    wv='wvalid',wr='wready',data='wdata',strb='wstrb',last='wlast',bv='bvalid',br='bready',bid='bid',resp='bresp')
CONTROL_MAP=dict(aw='awaddr',av='awvalid',ar='awready',wd='wdata',ws='wstrb',wv='wvalid',wr='wready',bv='bvalid',br='bready',bs='bresp',
    ra='araddr',rv='arvalid',rr='arready',ov='rvalid',ore='rready',od='rdata',os='rresp')


def sample(text):
    header,body=text.split('$enddefinitions $end',1)
    extra={'/board/EP/m_axib_awlock':1,**{'/board/EP/m_axil_'+k:w for k,w in CONTROL_WIDTHS.items()},
           **{ROOT+k:1 for k in ['host_rst_n','transport_rst_n','source_clk','lifetime_reset_n']}}
    extra.update(EXTRA)
    names={};widths={};scope=[];keep_header=[]
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            n='/'+ '/'.join(scope+[f[4]])
            require(n not in names,'unique VCD variables');names[n]=f[3];widths[f[3]]=int(f[2])
            if n in extra:require(widths[f[3]]==extra[n],'extra pin width');keep=False
        if keep:keep_header.append(line)
    require(set(extra).issubset(names) and len(names)==84,'exact84 native pins')
    native_codes={c for n,c in names.items() if n not in extra}
    decode=lambda line:line[1:].split() if line.startswith('b') else (line[0],line[1:])
    filtered=[]
    for line in body.splitlines():
        clean=line.strip()
        if not clean or clean[0] in '$#' or decode(clean)[1] in native_codes:filtered.append(line)
    native=sample_vcd('\n'.join(keep_header)+'\n$enddefinitions $end\n'+'\n'.join(filtered))
    require(set(names)==set(native['widths'])|set(extra),'complete VCD inventory')
    values={};updates={};now=0;host=[];source=[]
    def raw(n):return values.get(names[n],'x'*widths[names[n]])
    def flush():
        if raw('/board/EP/user_clk')=='0' and updates.get(names['/board/EP/user_clk'])=='1':
            host.append(dict(ps=now,reset=raw(ROOT+'host_rst_n'),ep_reset=raw('/board/EP/user_resetn'),
                data={k:raw('/board/EP/m_axib_'+v) for k,v in DATA_MAP.items()},
                control={k:raw('/board/EP/m_axil_'+v) for k,v in CONTROL_MAP.items()}))
        if raw(ROOT+'source_clk')=='0' and updates.get(names[ROOT+'source_clk'])=='1':
            source.append(dict(ps=now,host=raw(ROOT+'host_rst_n'),transport=raw(ROOT+'transport_rst_n'),lifetime=raw(ROOT+'lifetime_reset_n')))
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line[0]=='$':continue
        if line[0]=='#':
            flush();updates={};n=int(line[1:]);require(n>=now,'VCD chronology');now=n
        else:
            value,code=decode(line);value=value.lower()
            require(code in widths and set(value)<=set('01xz') and len(value)<=widths[code],'VCD encoding')
            updates[code]=value.rjust(widths[code],value[0] if value[0] in 'xz' else '0')
    flush()
    return native,host,source


def match_samples(sim,native,host,source,finish):
    def rows(prefix):return [fields(x[len(prefix):]) for x in sim.splitlines() if x.startswith(prefix)]
    hd=rows('NTRD_HOST_DATA ');hc=rows('NTRD_HOST_CTL ');reset=rows('NTRD_HOST_RESET ')
    require([int(x['ps']) for x in reset]==[x['ps'] for x in source],'source trace/VCD clock coverage')
    for a,z in zip(reset,source):
        require(all(a[k]==z[k] for k in ('host','transport','lifetime')),'actual reset pins match source trace')
    # A finish on a host rising edge may leave a final inactive sample in VCD.
    require([int(x['ps']) for x in hd]==[x['ps'] for x in host if x['ps']<=int(hd[-1]['ps'])],'all host clock samples witnessed')
    require(all(x['ps']==finish for x in host[len(hd):]),'no missing host samples before finish')
    for rows_,name,channels in [(hd,'data',DATA_CHANNELS),(hc,'control',CTL_CHANNELS)]:
        for a,z in zip(rows_,host):
            require(a['reset']==z['reset'],'same actual host reset')
            if z['reset']!='1':
                for k in (['ar','wr'] if name=='data' else ['ar','wr','rr']):
                    require(a[k]==z[name][k]=='0','actual request admission closed during reset')
                continue
            require(z['ep_reset']=='1','endpoint reset cannot drop in cold session')
            for kind,v,r,keys in channels:
                require(a[v]==z[name][v] and a[r]==z[name][r],'actual valid/ready pin values')
                if a[v]=='1':require(all(int(a[k],16)==bits(z[name][k]) for k in keys.split()),'actual active payload pin values')
    for z in host[len(hd):]:
        require(all(z['data'][k]=='0' for k in ['av','wv','bv']) and all(z['control'][k]=='0' for k in ['av','wv','bv','rv','ov']),'no unobserved final transactions')
    he,stalls=events(hc,CTL_CHANNELS,[x['reset']=='1' for x in hc])
    return he,stalls


def fixture():
    frames=[];writes=[(0xc4,1,3),(4,0,2),(0x14,1,0)]
    reads=[(4,0x10000),(8,0xff),(0xbc,2),(0xc0,37),(0x18,18),(0x1c,18),(0xb4,1),(0xb8,8)]
    for j in range(18):
        epoch=0x100000001+j;tag=0x800000010000abcd
        for i,value in enumerate([epoch,tag,0x1000fffc0,address(j)]):
            writes.extend([(0x20+8*i,value&0xffffffff,0),(0x24+8*i,value>>32,0)])
        writes.extend([(0xb0,j%2,0),(0x10,1,0)])
        for half,conflict in ([(0,0)] if j==12 else [(0,0),(0,1)] if j==13 else [(0,0),(1,0)]):
            payload=pattern(j)^conflict
            frame=b''.join(x.to_bytes(8,'little') for x in [0x313046555052544e,epoch,tag,half*32])
            frame+=((payload>>(half*256))&((1<<256)-1)).to_bytes(32,'little')
            frames.append(dict(job=j,frame=frame,resp=2 if conflict else 0))
        if j in (12,15,16):writes.append((0x10,2,0))
        writes.extend([(0x40,epoch&0xffffffff,0),(0x44,epoch>>32,0),(0x48,tag&0xffffffff,0),(0x4c,tag>>32,0),
                       (0x10,4,2),(0x10,4,0)])
    require((len(frames),len(writes),len(reads))==(35,294,8),'independent native fixture')
    return frames,writes,reads


def review(sim,tx_text,rx_text,vcd):
    checked=transport_review(sim)
    native,host,source=sample(vcd)
    def rows(prefix):return [fields(x[len(prefix):]) for x in sim.splitlines() if x.startswith(prefix)]
    enums=rows('NTRPU_LINK_ENUM ');ends=rows('NTRD_NATIVE_PASS ')
    require(len(enums)==len(ends)==1,'one enumeration and native summary')
    enum=enums[0];finish=int(ends[0]['ps']);start=int(enum['ps'])
    require({k:int(v) for k,v in ends[0].items()}==dict(writes=294,reads=8,frames=35,ps=finish),'native terminal inventory')
    require(native['end_ps']==finish==int(rows('NTRD_HOST_PASS ')[0]['ps']),'native/core finish agreement')
    require([enum['type'+str(i)] for i in range(3)]==['10','10','11'],'BAR types')
    require([int(enum['bar'+str(i)],16) for i in range(4)]==[0,0x10000,0,9] and int(enum['dma_bar'])==1,'BAR assignment')
    require(int(enum['range2'],16)&0xfffffff0==0xfff00000,'one MiB bypass BAR')
    control,_=match_samples(sim,native,host,source,finish)
    es=native['events'];after=[e for e in es if e['ps']>=start]
    groups={k:[e for e in after if e['channel']==k] for k in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require([len(groups[k]) for k in ['RQ','RC','TAG','AW','W','B','AR','R']]==[372,8,8,35,35,35,0,0],'complete native post-enumeration inventory')
    requests=groups['RQ'];packets=[];ctrl=[];i=0
    while i<len(requests):
        req=requests[i];a=bits(req['tdata'],63,0)
        if a==0x9000fffc0:
            require(i+1<len(requests) and bits(req['tlast'])==0 and bits(requests[i+1]['tlast'])==1,'two-beat native frame')
            packets.append((req,requests[i+1]));i+=2
        else:
            require(a<0x10000 and bits(req['tlast'])==1,'single control packet');ctrl.append(req);i+=1
    frames,writes,reads=fixture();require(len(packets)==35 and len(ctrl)==302,'native RQ partition')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start];rx=[p for p in packet_blocks(rx_text) if p['ps']>=start]
    require(len(tx)==337 and len(rx)==8,'decoded packet inventory')
    decoded={p['ps']:p for p in tx};require(len(decoded)==337,'unique decoded TX times')
    intents=rows('NTRD_NATIVE_FRAME ');require(len(intents)==35,'frame intent count')
    frame_metrics=[]
    for i,(fixture_,pair,h) in enumerate(zip(frames,packets,intents)):
        first,last=pair;d=first['tdata'];payload=fixture_['frame'];aw,w,b=[groups[k][i] for k in ['AW','W','B']]
        require(set(h)=={'n','job','addr','data','tag','ps'} and int(h['n'])==i+1 and int(h['job'])==fixture_['job'],'frame intent identity')
        require(int(h['addr'],16)==0x9000fffc0 and int(h['data'],16)==int.from_bytes(payload,'little') and int(h['tag'],16)==0x40+i,'frame intent bytes')
        require(bits(d,63,0)==0x9000fffc0 and bits(d,74,64)==16 and bits(d,78,75)==1 and bits(d,79,79)==0 and bits(d,127,120)==0,'frame RQ address/type/flags')
        require(bits(d,95,80)==0xaf and bits(d,119,104)==0x1a0 and bits(d,103,96)==0x40+i,'frame RQ IDs')
        require(bits(first['tkeep'])==65535 and bits(last['tkeep'])==15,'frame byte enables')
        require(bits(first['tuser'],15,0)==0xf0f and bits(first['tuser'],25,16)==16 and bits(first['tuser'],35,26)==0,'frame SOP')
        require(bits(last['tuser'],15,0)==0xf0f and bits(last['tuser'],25,16)==0 and bits(last['tuser'],35,26)==13,'frame EOP')
        require(bits(d,511,128).to_bytes(48,'little')+bits(last['tdata'],127,0).to_bytes(16,'little')==payload and bits(last['tdata'],511,128)==0,'all64 TLP data bytes')
        packet=decoded[last['ps']];v=packet['values']
        require(packet['kind']=='Memory Write-64 Frame' and packet['payload']==payload,'decoded frame payload')
        require((v['Address High']<<32)|v['Address Low']==0x9000fffc0 and v['Length']==16 and v['Tag']==0x40+i and v['Last and First Byte Enables']==255 and v['Requester Id']==0xaf,'decoded frame metadata')
        require([bits(aw[k]) for k in ['addr','id','len','size','burst']]==[0x1000fffc0,0,0,6,1],'native frame AXI address/attributes')
        require(bits(w['data'])==int.from_bytes(payload,'little') and bits(w['strb'])==(1<<64)-1 and bits(w['last'])==1,'native frame AXI data')
        require(bits(b['id'])==0 and bits(b['resp'])==fixture_['resp'],'native frame verdict')
        require(start<=int(h['ps'])<=first['ps']<last['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<finish,'native frame chronology')
        frame_metrics.append(dict(job=fixture_['job'],rq_to_b_ps=b['ps']-first['ps'],resp=fixture_['resp']))
    # Every data-pin event is also sampled in the independent transport trace.
    data_rows=rows('NTRD_HOST_DATA ');data_events,_=events(data_rows,DATA_CHANNELS,[x['reset']=='1' for x in data_rows])
    for ch,keys in [('AW','addr id len size burst lock'),('W','data strb last'),('B','id resp')]:
        for a,z in zip(data_events[ch],groups[ch]):
            payload=tuple(bits(z[k]) if k!='lock' else 0 for k in keys.split())
            require(a['ps']==z['ps'] and a['payload']==payload,'VCD and transport event identity')
    # Preserve and verify the public example's identification probe.
    probes=[e for e in es if e['channel']=='RQ' and e['ps']<start and bits(e['tdata'],78,75)==0 and bits(e['tdata'],63,0)==0]
    require(len(probes)==1 and len(control['AR'])==len(control['R'])==9,'one discovery read before8 fixture reads')
    probe=probes[0];par=control['AR'][0];pr=control['R'][0]
    require(par['payload']==(0,) and pr['payload']==(0x4452544e,0),'native NTRD identification')
    tags=[e for e in es if e['channel']=='TAG' and probe['ps']<=e['ps']<par['ps']]
    require(len(tags)==1,'discovery tag')
    completions=[e for e in es if e['channel']=='RC' and pr['ps']<=e['ps']<start and bits(e['tdata'],71,64)==tags[0]['tag']]
    require(len(completions)==1,'discovery completion')
    require(probe['ps']<par['ps']<pr['ps']<=completions[0]['ps']<start,'discovery ordering')
    require(bits(probe['tdata'],74,64)==1 and bits(probe['tdata'],127,120)==0x81 and bits(probe['tkeep'])==15 and bits(probe['tlast'])==1 and bits(probe['tuser'],15,0)==15,'discovery request framing')
    tx_probe=[p for p in packet_blocks(tx_text) if p['ps']==probe['ps']]
    rx_probe=[p for p in packet_blocks(rx_text) if p['ps']==completions[0]['ps']]
    require(len(tx_probe)==len(rx_probe)==1 and tx_probe[0]['kind']=='Memory Read-32 Frame' and tx_probe[0]['payload']==b'','decoded discovery request')
    request_groups={kind:[e for e in ctrl if bits(e['tdata'],78,75)==kind] for kind in [0,1]}
    require(len(request_groups[1])==294 and len(request_groups[0])==8,'control packet kinds')
    metrics=[]
    for kind,expected in [(1,writes),(0,reads)]:
        intent=rows('NTRPU_NATIVE_CONTROL_'+('WRITE ' if kind else 'READ '))
        require(len(intent)==len(expected),'control intent count')
        for i,(item,req,h) in enumerate(zip(expected,request_groups[kind],intent)):
            offset,value=item[:2];d=req['tdata']
            require(int(h['n'])==i+1 and int(h['addr'],16)==offset and int(h['data' if kind else 'expected'],16)==value,'control fixture intent')
            require(bits(d,63,0)==offset and bits(d,74,64)==1 and bits(d,79,79)==0 and bits(d,127,120)==0,'control RQ shape')
            require(bits(d,95,80)==0xaf and bits(d,119,104)==0x1a0 and bits(d,103,96)==int(h['tag'],16)==(0x20 if kind else 0x21),'control RQ identity')
            require(bits(req['tkeep'])==(31 if kind else 15) and bits(req['tuser'],15,0)==15,'control RQ framing')
            packet=decoded[req['ps']];v=packet['values']
            require(packet['kind']==('Memory Write-64 Frame' if kind else 'Memory Read-64 Frame') and packet['payload']==(value.to_bytes(4,'little') if kind else b''),'decoded control bytes')
            require((v['Address High']<<32)|v['Address Low']==offset and v['Tag']==int(h['tag'],16) and v['Length']==1 and v['Last and First Byte Enables']==15 and v['Requester Id']==0xaf,'decoded control header')
            require(start<=int(h['ps'])<=req['ps']<finish,'control stimulus chronology')
            if kind:
                aw,w,b=[control[k][i] for k in ['AW','W','B']]
                require(aw['payload']==(offset,) and w['payload']==(value,15) and b['payload']==(item[2],) and bits(d,159,128)==value,'control actual pins/verdict')
                require(req['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<finish,'control write ordering')
                metrics.append(dict(kind='write',offset=offset,rq_to_b_ps=b['ps']-req['ps'],resp=item[2]))
            else:
                ar,r=[control[k][i+1] for k in ['AR','R']]
                require(ar['payload']==(offset,) and r['payload']==(value,0) and int(h['mask'],16)==0xffffffff,'control read actual pins')
                rc=groups['RC'][i];tag=groups['TAG'][i]
                require(req['ps']<=tag['ps']<rc['ps'] and req['ps']<=ar['ps']<r['ps']<=rc['ps']<finish,'native read chain')
                metrics.append(dict(kind='read',offset=offset,rq_to_rc_ps=rc['ps']-req['ps']))
    # Account for all9 read completions, including the probe and decoded logs.
    values=rows('NTRPU_NATIVE_CONTROL_VALUE ');require(len(values)==8,'returned root values')
    for i,(offset,value,rc,tag,packet) in enumerate([(0,0x4452544e,completions[0],tags[0],rx_probe[0])]+
        [(offset,value,groups['RC'][i],groups['TAG'][i],rx[i]) for i,(offset,value) in enumerate(reads)]):
        d=rc['tdata'];v=packet['values']
        require(bits(d,15,0)==offset%4096 and bits(d,28,16)==4 and bits(d,30,29)==2 and bits(d,46,32)==1,'completion address/count/status')
        require(bits(d,71,64)==tag['tag'] and bits(d,63,48)==0xaf and bits(d,87,72)==0x1a0 and bits(d,94,89)==0 and bits(d,127,96)==value,'completion identity/data')
        require(bits(rc['tkeep'])==15 and bits(rc['tlast'])==1 and bits(rc['tuser'],67,0)==0x1000000000000f000 and bits(rc['tuser'],96,96)==0,'completion framing')
        require(packet['ps']==rc['ps'] and packet['kind']=='Completion With Data Frame' and v['Tag']==tag['tag'] and v['Completion Status']==0 and v['Length']==1 and v['Requester Id']==0xaf and v['Completer Id']==0x1a0 and packet['payload']==value.to_bytes(4,'big'),'decoded completion')
        if i:
            h=values[i-1];require(int(h['n'])==i and int(h['data'],16)==value and rc['ps']<=int(h['ps'])<finish,'root returned data')
    guard=review_startup(sim,vcd,host,source,finish)
    return dict(status='PASS_GUARDED_NATIVE_NTRD_DUAL_MIG_COLD',check=checked,guard=guard,signals=84,posted_frames=35,posted_control_writes=294,
        control_reads=9,discovery_reads=1,write_bytes=1024,readback_bytes=1792,final_ps=finish,
        frame_latencies=frame_metrics,control_latencies=metrics,native_stalls=native['stalls'],
        scope='Actual native XDMA root/endpoint and two host CDCs into retained NTRD/pool/two MIG generic DDR4 models.18 exclusive leases, rejected unfenced ACK, STOP/conflict and direct UI readback. Automatic guard uses actual native link/reset inputs and cold peer observations. External data barriers remain trusted BFMs; runtime reset coordination is unqualified. No physical PCIe/DDR, commercial DIMM, runtime link recovery, MSI crossing, routed timing, sustained throughput, P2P coherence or inference qualification.')
