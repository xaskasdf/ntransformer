"""Native BAR0 controls, BAR2 data and actual cold MIG, checked from pin traces.

The independent data component accounts for every data request and memory
transaction. This component accounts for all other requests, assigned tags,
completions, both AXI-Lite domains and root-port returned values. The barrier
remains a trusted test driver, not proof of actual upstream quiescence.
"""
from collections import defaultdict
from functools import lru_cache
import re
from check_ntrpu_pcie_bypass_native import bits,require
from check_ntrpu_pcie_bypass_smoke import fields,packet_blocks
from check_ntrpu_pcie_native_host_memory import review as memory_review,sample,CONTROL_WIDTHS,ROOT


def fixture():
    writes=[(0x14,1,0)]
    reads=[(0,0x4952544e,0xffffffff),(4,0x10000,0xffffffff),(8,1,0xffffffff)]
    for job in range(4):
        window=(1<<32)+[0,0x80000,0x80000,0xfffc0][job]
        destination=[0,0x100000040,0x100000040,0x1fffffffc0][job]
        for i,value in enumerate([job+1,0x2000+job,window,destination]):
            writes.extend([(0x20+8*i,value&0xffffffff,0),(0x24+8*i,value>>32,0)])
        writes.extend([(0x10,1,0),(0x20,0,2),(0x10,4,2)])
        reads.extend([(0x18,job,0xffffffff),(0x18,job+1,0xffffffff)])
        if job==2:writes.append((0x10,2,0));reads.append((0x0c,0x10,0x10))
        reads.extend([(0x60,job+1,0xffffffff),(0x64,0,0xffffffff),(0x68,0x2000+job,0xffffffff),
                      (0x6c,0,0xffffffff),(0x70,1 if job==2 else 0,0xffffffff),(0x0c,1,1)])
        writes.extend([(0x40,job+1,0),(0x44,0,0),(0x48,0x2000+job,0),(0x4c,1,0),
                       (0x10,4,2),(0x4c,0,0),(0x10,4,0),(0x10,1,2)])
        reads.extend([(0x1c,job,0xffffffff),(0x1c,job+1,0xffffffff),(0x0c,0,1),(0x18,job+1,0xffffffff)])
    require(len(writes)==78 and len(reads)==52,'control fixture sizes')
    return writes,reads


@lru_cache(maxsize=1)
def sample_control(text):
    header,body=text.split('$enddefinitions $end',1)
    names={};widths={};scope=[]
    for line in header.splitlines():
        f=line.split()
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+ '/'.join(scope+[f[4]])
            require(name not in names,'duplicate VCD name');names[name]=f[3];widths[f[3]]=int(f[2])
    channels={'AW':['awaddr'],'W':['wdata','wstrb'],'B':['bresp'],'AR':['araddr'],'R':['rdata','rresp']}
    domains={'HOST':('/board/EP/m_axil_','/board/EP/user_clk',ROOT+'host_rst_n'),
             'DEVICE':(ROOT+'dut/d_axil_',ROOT+'clk',ROOT+'rst_n')}
    for prefix,clock,reset in domains.values():
        require(clock in names and reset in names,'control clocks/resets')
        for k,w in CONTROL_WIDTHS.items():require(prefix+k in names and widths[names[prefix+k]]==w,'control width '+k)
    values={};updates={};held={};events=[];stalls=defaultdict(int);now=0
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        for domain,(prefix,clock,reset) in domains.items():
            if raw(clock)!='0' or updates.get(names[clock])!='1' or raw(reset)!='1':continue
            for channel,keys in channels.items():
                ch=channel.lower();valid=raw(prefix+ch+'valid');ready=raw(prefix+ch+'ready')
                payload={k:raw(prefix+k) for k in keys};key=domain+'_'+channel
                if key in held:require(valid=='1' and payload==held.pop(key),'unstable control '+key)
                if valid=='1':
                    require(ready in {'0','1'},'unknown control ready')
                    if ready=='1':events.append(dict(domain=domain,channel=channel,ps=now,**{k:bits(v) for k,v in payload.items()}))
                    else:held[key]=payload;stalls[key]+=1
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            flush();updates={};new=int(line[1:]);require(new>=now,'control VCD chronology');now=new
        else:
            value,code=line[1:].split() if line.startswith('b') else (line[0],line[1:])
            require(code in widths and set(value)<=set('01xz') and len(value)<=widths[code],'control VCD encoding')
            updates[code]=value.rjust(widths[code],value[0] if value[0] in 'xz' else '0')
    flush();require(not held,'control final drain')
    return dict(events=events,stalls=dict(stalls),end_ps=now)


def review(sim,tx_text,rx_text,vcd):
    result=memory_review(sim,tx_text,rx_text,vcd)
    native,aux=sample(vcd);control=sample_control(vcd)
    enum=fields(next(l for l in sim.splitlines() if l.startswith('NTRPU_LINK_ENUM ')))
    start=int(enum['ps']);finish=result['final_ps'];base=int(enum['bar0'],16)
    require(base==0 and int(enum['bar1'],16)==0x10000,'public root-port BAR0/BAR1 assignment')
    rows=defaultdict(list)
    for line in sim.splitlines():
        if line.startswith('NTRPU_NATIVE_CONTROL_'):
            kind=line.split()[0].removeprefix('NTRPU_NATIVE_CONTROL_');rows[kind].append(fields(line))
    expected_keys={'WRITE','READ','VALUE','PASS'}|{d+'_'+c for d in ['HOST','DEVICE'] for c in ['AW','W','B','AR','R']}
    require(set(rows)==expected_keys,'control marker schema')
    require(len(rows['PASS'])==1 and {k:int(v) for k,v in rows['PASS'][0].items()}==dict(writes=78,reads=52,data_writes=83,ps=finish),'control summary')
    writes,reads=fixture()
    require([len(rows[k]) for k in ['WRITE','READ','VALUE']]==[78,52,52],'control host intent count')
    all_sampled={d+'_'+c:[e for e in control['events'] if e['domain']==d and e['channel']==c]
             for d in ['HOST','DEVICE'] for c in ['AW','W','B','AR','R']}
    for key,events in all_sampled.items():
        require(len(events)==(78 if key.split('_')[1] in {'AW','W','B'} else 53),'control pin event count '+key)
        require(len(rows[key])==len(events),'control event/log count '+key)
        for actual,logged in zip(events,rows[key]):
            require(set(logged)==set(actual)-{'domain','channel'},'control log fields')
            require({k:int(v,10 if k=='ps' else 16) for k,v in logged.items()}=={k:v for k,v in actual.items() if k not in {'domain','channel'}},'control log/pin mismatch')
    sampled={k:[e for e in es if e['ps']>=start] for k,es in all_sampled.items()}
    discovery_pins={k:[e for e in es if e['ps']<start] for k,es in all_sampled.items()}
    for key,events in discovery_pins.items():
        require(len(events)==int(key.endswith(('_AR','_R'))),'one read-only discovery transfer '+key)
    # The unchanged AMD XDMA-bar search reads BAR0 once before our fixture.
    # Preserve that request, assigned tag and completion instead of dropping it.
    probes=[e for e in native['events'] if e['channel']=='RQ' and e['ps']<start and bits(e['tdata'],78,75)==0 and bits(e['tdata'],63,0)==base]
    require(len(probes)==1,'one native BAR0 discovery read')
    probe=probes[0];p_ar=discovery_pins['HOST_AR'][0];p_r=discovery_pins['HOST_R'][0]
    require(p_ar['araddr']==0 and p_r['rdata']==0x4952544e and p_r['rresp']==0,'discovery ID read')
    tags=[e for e in native['events'] if e['channel']=='TAG' and probe['ps']<=e['ps']<p_ar['ps']]
    require(len(tags)==1,'one allocated discovery tag')
    completions=[e for e in native['events'] if e['channel']=='RC' and p_r['ps']<=e['ps']<start and bits(e['tdata'],71,64)==tags[0]['tag']]
    require(len(completions)==1,'one native discovery completion')
    p_rc=completions[0];pd=p_rc['tdata']
    require(probe['ps']<p_ar['ps']<p_r['ps']<=p_rc['ps']<start,'discovery chronology')
    require(bits(probe['tdata'],74,64)==1 and bits(probe['tkeep'])==15 and bits(probe['tlast'])==1 and bits(probe['tuser'],15,0)==15,'discovery request framing')
    require(bits(probe['tdata'],79,79)==0 and bits(probe['tdata'],95,80)==0xaf and bits(probe['tdata'],119,104)==0x1a0 and bits(probe['tdata'],127,120)==0x81,'public discovery request IDs/flags')
    require(bits(pd,15,0)==0 and bits(pd,28,16)==4 and bits(pd,30,29)==2 and bits(pd,46,32)==1 and bits(pd,127,96)==0x4952544e,'discovery completion fields')
    require(bits(pd,63,48)==0xaf and bits(pd,87,72)==0x1a0 and bits(pd,94,89)==0,'discovery completion IDs/attributes')
    require(bits(p_rc['tkeep'])==15 and bits(p_rc['tlast'])==1 and bits(p_rc['tuser'],67,0)==0x1000000000000f000 and bits(p_rc['tuser'],96,96)==0,'discovery completion framing')
    p_tx=[p for p in packet_blocks(tx_text) if p['ps']==probe['ps']]
    p_rx=[p for p in packet_blocks(rx_text) if p['ps']==p_rc['ps']]
    require(len(p_tx)==len(p_rx)==1,'discovery decoded packets')
    require(p_tx[0]['kind']=='Memory Read-32 Frame' and p_tx[0]['values']['Tag']==bits(probe['tdata'],103,96) and p_tx[0]['payload']==b'','discovery decoded request')
    require(p_rx[0]['kind']=='Completion With Data Frame' and p_rx[0]['values']['Tag']==tags[0]['tag'] and p_rx[0]['values']['Completion Status']==0 and p_rx[0]['payload']==bytes.fromhex('4952544e'),'discovery decoded completion')
    require(all(p_tx[0]['values'][k]==v for k,v in {'Address Low':0,'Length':1,'Requester Id':0xaf,'Last and First Byte Enables':15}.items()),'discovery decoded request fields')
    require(all(p_rx[0]['values'][k]==v for k,v in {'Length':1,'Requester Id':0xaf,'Completer Id':0x1a0}.items()),'discovery decoded completion fields')
    native_events=[e for e in native['events'] if e['ps']>=start]
    requests=[e for e in native_events if e['channel']=='RQ']
    require(len(requests)==213,'all native RQ packets accounted')
    data_rq=[e for e in requests if (9<<32)<=bits(e['tdata'],63,0)<(9<<32)+(1<<20)]
    ctrl_rq=[e for e in requests if base<=bits(e['tdata'],63,0)<base+(1<<16)]
    require(len(data_rq)==83 and len(ctrl_rq)==130,'disjoint native address windows')
    groups={ch:[e for e in native_events if e['channel']==ch] for ch in ['RC','TAG']}
    require(len(groups['RC'])==len(groups['TAG'])==52,'all completion/tag counts')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start];rx=[p for p in packet_blocks(rx_text) if p['ps']>=start]
    require(len(tx)==213 and len(rx)==52,'all decoded packet counts')
    decoded={p['ps']:p for p in tx};require(len(decoded)==213,'unique transmitted packet times')
    rq_by_kind={kind:[e for e in ctrl_rq if bits(e['tdata'],78,75)==kind] for kind in [0,1]}
    require(len(rq_by_kind[1])==78 and len(rq_by_kind[0])==52,'control request types')
    measurements=[]
    for kind,expected in [(1,writes),(0,reads)]:
        for i,(offset,value,last) in enumerate(expected):
            req=rq_by_kind[kind][i];data=req['tdata'];word=value if kind else None
            host=rows['WRITE' if kind else 'READ'][i]
            schema={'n','addr','data','tag','ps'} if kind else {'n','addr','expected','mask','tag','ps'}
            require(set(host)==schema and int(host['n'])==i+1,'host control intent schema/sequence')
            require(int(host['addr'],16)==bits(data,63,0)==base+offset,'control request address')
            require(int(host['data' if kind else 'expected'],16)==value,'control intent value')
            if not kind:require(int(host['mask'],16)==last,'control read mask')
            require(bits(data,74,64)==1 and bits(data,79,79)==0 and bits(data,127,120)==0,'control RQ attributes')
            require(bits(data,95,80)==0xaf and bits(data,119,104)==0x1a0,'control RQ IDs')
            require(bits(data,103,96)==int(host['tag'],16)==(0x20 if kind else 0x21),'control requested tag')
            require(bits(req['tuser'],7,0)==15 and bits(req['tuser'],15,8)==0,'control byte enables')
            require(bits(req['tkeep'])==(31 if kind else 15) and bits(req['tlast'])==1,'control packet framing')
            packet=decoded[req['ps']];v=packet['values']
            require(packet['kind']==('Memory Write-64 Frame' if kind else 'Memory Read-64 Frame'),'decoded control kind')
            require((v['Address High']<<32)|v['Address Low']==base+offset and v['Tag']==int(host['tag'],16),'decoded control address/tag')
            require(v['Length']==1 and v['Last and First Byte Enables']==15,'decoded control attributes')
            require(packet['payload']==(word.to_bytes(4,'little') if kind else b''),'decoded control payload')
            require(start<=int(host['ps'])<=req['ps']<finish,'host intent chronology')
            if kind:
                aw,w,b=(sampled['HOST_'+c][i] for c in ['AW','W','B'])
                require(bits(data,159,128)==value and aw['awaddr']==offset and w['wdata']==value and w['wstrb']==15 and b['bresp']==last,'control write pin values')
                require(req['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<finish,'control write chronology')
                measurements.append(dict(kind='write',offset=offset,request_ps=req['ps'],response_ps=b['ps'],bresp=last))
            else:
                ar,r=(sampled['HOST_'+c][i] for c in ['AR','R']);rc=groups['RC'][i];tag=groups['TAG'][i];d=rc['tdata']
                require(ar['araddr']==offset and r['rresp']==0 and r['rdata']&last==value,'control read pin value')
                require(bits(d,11,0)==offset%4096 and bits(d,15,12)==0,'control RC address/error')
                require(bits(d,28,16)==4 and bits(d,29,29)==0 and bits(d,30,30)==1,'control RC byte count/completed')
                require(bits(d,42,32)==1 and bits(d,46,43)==0,'control RC count/status')
                require(bits(d,71,64)==tag['tag'] and bits(d,63,48)==0xaf and bits(d,87,72)==0x1a0,'control RC IDs/assigned tag')
                require(bits(d,94,89)==0 and bits(d,127,96)==r['rdata'],'control completion data')
                require(bits(rc['tkeep'])==15 and bits(rc['tlast'])==1,'control RC framing')
                require(bits(rc['tuser'],63,0)==0xf000 and bits(rc['tuser'],67,64)==1 and bits(rc['tuser'],96,96)==0,'control RC attributes')
                returned=rows['VALUE'][i];require(set(returned)=={'n','data','ps'} and int(returned['n'])==i+1 and int(returned['data'],16)==r['rdata'],'root returned value')
                packet=rx[i];v=packet['values']
                require(packet['kind']=='Completion With Data Frame' and packet['ps']==rc['ps'],'decoded control completion/time')
                require(v['Tag']==tag['tag'] and v['Length']==1 and v['Completion Status']==0 and packet['payload']==r['rdata'].to_bytes(4,'big'),'decoded completion identity/payload')
                require(req['ps']<=tag['ps']<rc['ps']<=int(returned['ps'])<finish and req['ps']<=ar['ps']<r['ps']<=rc['ps'],'control read causal chain')
                measurements.append(dict(kind='read',offset=offset,request_ps=req['ps'],response_ps=rc['ps'],allocated_tag=tag['tag'],value=r['rdata']))
    delays={}
    for channel in ['AW','W','B','AR','R']:
        samples=[]
        for host,device in zip(all_sampled['HOST_'+channel],all_sampled['DEVICE_'+channel]):
            payload=lambda e:{k:v for k,v in e.items() if k not in {'domain','channel','ps'}}
            require(payload(host)==payload(device),'AXIL CDC payload')
            require(device['ps']>host['ps'] if channel in {'B','R'} else host['ps']>device['ps'],'AMD AXIL acknowledgement order')
            require(0<min(host['ps'],device['ps']) and max(host['ps'],device['ps'])<finish,'control transfer drain')
            samples.append(abs(host['ps']-device['ps']))
        delays[channel]=dict(min_ps=min(samples),max_ps=max(samples),samples=len(samples),measurement='receiver_handshake_to_source_acknowledgement')
    require(control['end_ps']==finish,'control witness end time')
    result.update(status='PASS_NATIVE_PCIE_CONTROL_DATA_MIG_PERIODIC',control=dict(writes=78,reads=52,rejected_writes=16,
        requests=130,completions=52,assigned_tags=52,discovery_reads=1,total_control_reads=53,bar_base=base,
        discovery=dict(request_ps=probe['ps'],response_ps=p_rc['ps'],allocated_tag=tags[0]['tag'],value=p_r['rdata']),
        acknowledgement_delays=delays,stalls=control['stalls'],measurements=measurements),
        scope='Native root-port BAR0 control and BAR2 DWORD writes, two actual AMD CDCs, unchanged UI controller/guarded ingress and cold generic MIG. Four reservations, early/bad ACK, descriptor lock, stale ARM, conflict/STOP/recovery and memory readback. External producer isolation and direct UI readback remain BFMs; no actual upstream fence, physical CDC/timing, exact DIMM, warm reset, MSI integration, board/P2P or inference qualification.')
    return result
