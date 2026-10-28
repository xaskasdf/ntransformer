# Data-path component only. The paired control checker must account for every
# additional RQ/RC/TAG event and AXI-Lite transfer in this 184-signal bank.
"""Sample native PCIe plus both AXI domains and compare actual MIG readback.

This is the bounded four-job fixture, not a general protocol checker or a
physical CDC proof. AMD FIFO behavioral synchronization-delay limits apply.
"""
import re
from functools import lru_cache
from check_ntrpu_pcie_bypass_native import bits, require, sample_vcd
from check_ntrpu_pcie_bypass_smoke import fields, packet_blocks
from check_ntrpu_pcie_line_link import ROOT, WIDTHS as INGRESS_WIDTHS
from check_ntrpu_pcie_ingress_peer_link import WIDTHS as PEER_WIDTHS

from check_ntrpu_pcie_fragment_native_frames import fixture,review_frames

CONTROL_WIDTHS=dict(awaddr=32,wdata=32,araddr=32,wstrb=4,awvalid=1,wvalid=1,bready=1,arvalid=1,rready=1,awready=1,wready=1,arready=1,bvalid=1,rvalid=1,bresp=2,rresp=2,rdata=32)

DESTINATIONS=[0,0x100000040,0x100000040,0x1fffffffc0]
WRITE_JOBS=[0,1,3]
STATE_WIDTHS={**INGRESS_WIDTHS,**{k:v for k,v in PEER_WIDTHS.items() if not k.startswith('m_axi_')},
              'clk':1,'rst_n':1,'host_rst_n':1,'calibrated':1,'ui_rst':1,'guard_idle':1,'irq':1,'protocol_fault':1,'rejected_frames':32,'rejected_identity':32,'last_reject_code':4,'last_reject_faults':7}
MEMORY_WIDTHS={k:v for k,v in PEER_WIDTHS.items() if k.startswith('m_axi_')}
MEMORY_WIDTHS['m_axi_awaddr']=37
for suffix,width in {'araddr':37,'arid':4,'arlen':8,'arsize':3,'arburst':2,'arlock':1,'arcache':4,'arprot':3,
                     'arvalid':1,'arready':1,'rdata':512,'rid':4,'rresp':2,'rlast':1,'rvalid':1,'rready':1}.items():
    MEMORY_WIDTHS['m_axi_'+suffix]=width
DEVICE_WIDTHS={'awaddr':64,'awid':4,'awlen':8,'awsize':3,'awburst':2,'awlock':1,'awvalid':1,'awready':1,
               'wdata':512,'wstrb':64,'wlast':1,'wvalid':1,'wready':1,'bid':4,'bresp':2,'bvalid':1,'bready':1}


@lru_cache(maxsize=1)
def sample(text):
    header,body=text.split('$enddefinitions $end',1)
    extra={ROOT+k:v for k,v in {**STATE_WIDTHS,**MEMORY_WIDTHS}.items()}
    extra.update({ROOT+'dut/data_path/d_axi_'+k:v for k,v in DEVICE_WIDTHS.items()})
    extra['/board/EP/m_axib_awlock']=1
    extra.update({prefix+k:v for prefix in ['/board/EP/m_axil_',ROOT+'dut/d_axil_'] for k,v in CONTROL_WIDTHS.items()})
    names={};widths={};scope=[];native_header=[]
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+ '/'.join(scope+[f[4]])
            require(name not in names,'duplicate VCD variable')
            names[name]=f[3];widths[f[3]]=int(f[2])
            if name in extra:
                require(int(f[2])==extra[name],'extra VCD width '+name);keep=False
        if keep:native_header.append(line)
    require(set(extra).issubset(names),'missing native/MIG/CDC signals')
    native_codes={code for name,code in names.items() if name not in extra}
    def decode(line):return line[1:].split() if line.startswith('b') else (line[0],line[1:])
    filtered=[]
    for line in body.splitlines():
        clean=line.strip()
        if not clean or clean[0] in '$#' or decode(clean)[1] in native_codes:filtered.append(line)
    native=sample_vcd('\n'.join(native_header)+'\n$enddefinitions $end\n'+'\n'.join(filtered))
    require(set(names)==set(native['widths'])|set(extra) and len(names)==184,'exact 184-signal inventory')
    channels={c:(ROOT+p,fs) for c,p,fs in [
        ('ARM','arm_',['epoch','tag','window','destination']),('LINE','line_',['epoch','tag','address','data']),
        ('DONE','done_',['epoch','tag','error']),('RESULT','result_',['epoch','tag','status'])]}
    for c,p,fs in [('MAW','m_axi_aw',['addr','id','len','size','burst','cache','prot','lock']),
                   ('MW','m_axi_w',['data','strb','last']),('MB','m_axi_b',['id','resp']),
                   ('AR','m_axi_ar',['addr','id','len','size','burst','cache','prot','lock']),
                   ('R','m_axi_r',['id','data','resp','last']),
                   ('DAW','dut/data_path/d_axi_aw',['addr','id','len','size','burst','lock']),
                   ('DW','dut/data_path/d_axi_w',['data','strb','last']),('DB','dut/data_path/d_axi_b',['id','resp'])]:
        channels[c]=(ROOT+p,fs)
    values={};updates={};now=0;events=[];states={};held={};stalls={c:0 for c in channels};prior_ui=None;live=False;rq_open=False
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        nonlocal prior_ui,live,rq_open
        rp=names['/board/RP/user_clk']
        if values.get(rp)=='0' and updates.get(rp)=='1':
            valid=raw('/board/RP/s_axis_rq_tvalid');ready=raw('/board/RP/s_axis_rq_tready')
            if rq_open:require(valid=='1','RQ VALID dropped within packet')
            if valid=='1' and ready=='1':rq_open=not bool(bits(raw('/board/RP/s_axis_rq_tlast')))
        host=names['/board/EP/user_clk']
        if values.get(host)=='0' and updates.get(host)=='1' and raw('/board/EP/user_resetn')=='1':
            if raw('/board/EP/m_axib_awvalid')=='1':
                require(bits(raw('/board/EP/m_axib_awlock'))==0 and raw(ROOT+'host_rst_n')=='1','host AWLOCK/reset')
        clock=names[ROOT+'clk']
        if values.get(clock)=='0' and updates.get(clock)=='1':
            if live:require(raw(ROOT+'rst_n')=='1','runtime reset not qualified')
            if raw(ROOT+'rst_n')=='1':
                live=True
                if prior_ui is not None:require(now-prior_ui==5000,'actual UI period')
                prior_ui=now
                state={k:bits(raw(ROOT+k)) for k in STATE_WIDTHS};states[now]=state
                require(state['calibrated']==1 and state['ui_rst']==state['protocol_fault']==0 and (state['stop']==0 or (state['epoch']==0x100000003 and state['error_code'] in {0,1})),'cold MIG state')
                require(state['peer_epoch']==state['epoch'],'joint epoch')
                require(not (state['result_valid'] and (state['peer_axi_busy'] or not state['guard_idle'])),'result before memory drain')
                for ch,(prefix,fs) in channels.items():
                    payload={f:raw(prefix+f) for f in fs};valid=raw(prefix+'valid');ready=raw(prefix+'ready')
                    if ch in held:require(valid=='1' and payload==held.pop(ch),'unstable '+ch)
                    if valid=='1':
                        require(ready in {'0','1'},'unknown ready '+ch)
                        if ready=='1':
                            # Only inactive source W lanes may remain X.
                            e={f:(v if ch=='DW' and f=='data' else bits(v)) for f,v in payload.items()}
                            events.append(dict(channel=ch,ps=now,**e))
                        else:held[ch]=payload;stalls[ch]+=1
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            flush();updates={};new=int(line[1:]);require(new>=now,'VCD chronology');now=new
        else:
            value,code=decode(line);value=value.lower()
            require(code in widths and set(value)<=set('01xz') and len(value)<=widths[code],'VCD encoding')
            updates[code]=value.rjust(widths[code],value[0] if value[0] in 'xz' else '0')
    flush();require(live and not held and not rq_open,'unfinished UI or RQ transfer')
    return native,dict(events=events,states=states,stalls=stalls,end_ps=now,signals=len(names))


def review(sim,tx_text,rx_text,vcd, *, require_periodic=True):
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)|\b(?:VIOLATION|VIOLATED|INCORRECT_PARAMETER)\b|SIMULATIONS NOT SUPPORTED',sim,re.M|re.I),'simulation failure')
    events=[(s.split()[0].removeprefix('NTRPU_LINK_'),fields(s)) for s in sim.splitlines() if s.startswith('NTRPU_LINK_')]
    def rows(kind):return [f for k,f in events if k==kind]
    def one(kind):
        r=rows(kind);require(len(r)==1,'one '+kind);return r[0]
    enum,summary,final=one('ENUM'),one('COUNTS'),one('PASS')
    require([enum['type'+str(i)] for i in range(3)]==['10','10','11'],'BAR types')
    require(int(enum['bar2'],16)==0 and int(enum['bar3'],16)==9 and int(enum['dma_bar'])==1,'BAR assignment')
    require(int(enum['range2'],16)&0xfffffff0==0xfff00000,'BAR size')
    require([int(final[k]) for k in ['jobs','writes','lines','published_bytes']]==[4,12,3,192],'final contract')
    require([int(summary[k]) for k in ['aw','w','b','lines','results']]==[12,12,12,3,4],'logged counts')
    start,finish=int(enum['ps']),int(final['ps'])
    native,aux=sample(vcd)
    device={c:[e for e in aux['events'] if e['channel']==c] for c in ['DAW','DW','DB']}
    require(all(len(v)==12 for v in device.values()),'12 actual CDC output transfers')
    require(native['end_ps']==aux['end_ps']==finish==int(summary['ps']),'VCD final drain')
    require(all(int(summary[ch.lower()+'_stall'])==native['stalls'][ch] for ch in ['AW','W','B']),'native stalls')
    groups,device,fixtures,banks,bitmaps,measurements,accepted_bytes=review_frames(sim,tx_text,native,aux,rows,start,finish)
    auxgroups={ch:[e for e in aux['events'] if e['channel']==ch and e['ps']>=start] for ch in ['ARM','LINE','DONE','RESULT']}
    require([len(auxgroups[k]) for k in ['ARM','LINE','DONE','RESULT']]==[4,3,3,4],'ingress native event counts')
    for kind in auxgroups:require(len(rows(kind))==len(auxgroups[kind]),'ingress log count '+kind)
    for kind,jobs in [('ARM',[0,1,2,3]),('LINE',[0,1,3]),('RESULT',[0,1,2,3])]:
        require([int(x['job']) for x in rows(kind)]==jobs,'logged job sequence '+kind)
    for job,offset in enumerate([0,0x80000,0x80000,0xfffc0]):
        arm,result=auxgroups['ARM'][job],auxgroups['RESULT'][job]
        astate,rstate=aux['states'][arm['ps']],aux['states'][result['ps']]
        require(arm['epoch']==result['epoch']==0x100000001+job and arm['tag']==result['tag']==0x8000000100002000+job,'job identity')
        require(arm['window']==(1<<32)+offset and arm['destination']==DESTINATIONS[job],'job mapping')
        require(astate['source_stopped']==astate['ingress_isolated']==astate['transport_idle']==1 and astate['quiesce_epoch']==astate['epoch']==(0 if job==0 else 0x100000000+job),'arm barrier')
        require(rstate['source_stopped']==rstate['ingress_isolated']==rstate['transport_idle']==1 and rstate['quiesce_epoch']==0x100000001+job,'result barrier')
        require(result['status']==(1 if job==2 else 0),'job result')
        indices=[i for i,w in enumerate(fixtures) if w['job']==job]
        require(arm['ps']<groups['RQ'][indices[0]]['ps'] and groups['B'][indices[-1]]['ps']<result['ps'],'job transport drain')
        if job<3:require(result['ps']<auxgroups['ARM'][job+1]['ps'],'rearm after retirement')
        if job!=2:
            index=[0,1,3].index(job);line,done=auxgroups['LINE'][index],auxgroups['DONE'][index]
            require(line['epoch']==done['epoch']==0x100000001+job and line['tag']==done['tag']==0x8000000100002000+job,'destination identity')
            require(line['address']==DESTINATIONS[job] and line['data']==int.from_bytes(bytes(banks[job]),'little'),'assembled line bytes')
            require(done['error']==0 and device['DB'][indices[-1]]['ps']<line['ps']<done['ps']<result['ps'],'destination completion chain')
        else:require(bitmaps[job]==0xffffffff and rstate['failed']==1 and rstate['error_code']==1,'STOP retains half line without memory write')
    for kind,mapping in [('ARM',{'window':'window','dest':'destination','epoch':'epoch','tag':'tag'}),
                         ('LINE',{'addr':'address','data':'data','epoch':'epoch','tag':'tag'}),
                         ('DONE',{'epoch':'epoch','tag':'tag','error':'error'}),
                         ('RESULT',{'epoch':'epoch','tag':'tag','status':'status'})]:
        for logged,n in zip(rows(kind),auxgroups[kind]):
            require(int(logged['ps'])==n['ps'],'egress log time')
            require(all(int(logged[k],10 if k in {'status','error'} else 16)==n[v] for k,v in mapping.items()),'egress log fields')

    memory={c:[e for e in aux['events'] if e['channel']==c] for c in ['MAW','MW','MB','AR','R']}
    require([len(memory[c]) for c in memory]==[3,3,3,8,8],'actual MIG transfer counts')
    dest_logs=[(s.split()[0].removeprefix('NTRPU_PEER_LINK_'),fields(s)) for s in sim.splitlines() if s.startswith('NTRPU_PEER_LINK_')]
    require(len(dest_logs)==9 and {c for c,_ in dest_logs}=={'MAW','MW','MB'},'MIG write log schema')
    extra=[(s.split()[0].removeprefix('NTRPU_NATIVE_MIG_'),fields(s)) for s in sim.splitlines() if s.startswith('NTRPU_NATIVE_MIG_')]
    require({c for c,_ in extra}=={'START','STATE','PASS','DAW','DW','DB','AR','R'},'full-chain marker inventory')
    for ch in ['DAW','DW','DB']:
        logged=[f for c,f in extra if c==ch];require(len(logged)==12,'CDC log count '+ch)
        for row,event in zip(logged,device[ch]):
            require(set(row)==set(event)-{'channel'},'CDC log fields')
            for k,v in event.items():
                if k=='channel':continue
                if ch=='DW' and k=='data':
                    for lane in range(64):
                        if event['strb']>>lane&1:
                            require(int(row[k][128-2*(lane+1):128-2*lane],16)==bits(v,lane*8+7,lane*8),'CDC logged active byte')
                else:require(int(row[k],10 if k in {'ps','len','size','burst','lock','last','resp'} else 16)==v,'CDC log/native mismatch')
    committed={};read_cursor=0;memory_latencies=[]
    for job in range(4):
        arm,retired=auxgroups['ARM'][job],auxgroups['RESULT'][job]
        state=aux['states'][retired['ps']]
        require(state['peer_axi_busy']==state['bridge_phase']==state['destination_status']==0 and state['guard_idle']==1,'retired memory state')
        phases=[]
        for ps,st in aux['states'].items():
            if arm['ps']<=ps<=retired['ps']:
                phase=st['bridge_phase']
                if not phases or phases[-1]!=phase:phases.append(phase)
        require(phases==([0] if job==2 else [0,1,2,3,4,5,0]),'bridge phase path')
        if job in WRITE_JOBS:
            i=WRITE_JOBS.index(job);aw,w,b=[memory[c][i] for c in ['MAW','MW','MB']]
            line,done=auxgroups['LINE'][i],auxgroups['DONE'][i]
            require(aw['addr']==line['address']==DESTINATIONS[job] and aw['size']==6 and aw['burst']==1,'MIG address/size')
            require(all(aw[k]==0 for k in ['id','len','cache','prot','lock']),'MIG AW sidebands')
            require(w['data']==line['data'] and w['strb']==(1<<64)-1 and w['last']==1,'actual full MIG write bytes')
            require(b['id']==b['resp']==0,'MIG write response')
            require(line['ps']<min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<done['ps']<retired['ps'],'MIG causal completion')
            require(aux['states'][b['ps']]['peer_axi_busy']==1 and aux['states'][done['ps']]['peer_axi_busy']==0,'memory B releases receiver')
            committed[aw['addr']]=int.from_bytes(bytes(banks[job]),'little')
            for ch,event in [('MAW',aw),('MW',w),('MB',b)]:
                row=[f for c,f in dest_logs if c==ch][i]
                require(set(row)==set(event)-{'channel'},'MIG write log fields')
                for k,v in event.items():
                    if k!='channel':require(int(row[k],10 if k in {'ps','len','size','burst','lock','last','resp'} else 16)==v,'MIG log/native mismatch')
            memory_latencies.append(dict(job=job,line_to_b_ps=b['ps']-line['ps'],b_to_done_ps=done['ps']-b['ps']))
        else:
            require(not any(arm['ps']<=e['ps']<=retired['ps'] for c in ['MAW','MW','MB'] for e in memory[c]),'cancelled partial line reached MIG')
        for address,value in sorted(committed.items()):
            ar,r=memory['AR'][read_cursor],memory['R'][read_cursor]
            require(ar['addr']==address and ar['size']==6 and ar['burst']==1,'sentinel read address')
            require(all(ar[k]==0 for k in ['id','len','cache','prot','lock']),'sentinel read attributes')
            require(r['data']==value and r['id']==r['resp']==0 and r['last']==1,'sentinel read bytes/status')
            require(retired['ps']<ar['ps']<r['ps']<finish,'sentinel after retirement')
            if job<3:require(r['ps']<auxgroups['ARM'][job+1]['ps'],'sentinel before next rearm')
            for ch,event in [('AR',ar),('R',r)]:
                row=[f for c,f in extra if c==ch][read_cursor]
                expected_keys={'job','addr','ps'} if ch=='AR' else {'job','addr','data','id','resp','last','ps'}
                require(set(row)==expected_keys and int(row['job'])==job and int(row['addr'],16)==address,'sentinel log context')
                for k in expected_keys-{'job','addr'}:
                    require(int(row[k],10 if k in {'ps','resp','last'} else 16)==event[k],'sentinel log/native mismatch')
            read_cursor+=1
    require(read_cursor==8 and aux['stalls']['R']>=24,'read count/stability coverage')
    for ch in ['AR','R']:require(len([f for c,f in extra if c==ch])==8,'sentinel logs exact count')
    starts=[f for c,f in extra if c=='START'];ends=[f for c,f in extra if c=='PASS']
    require(len(starts)==len(ends)==1 and set(starts[0])=={'ps'},'memory start/final markers')
    require(set(ends[0])=={'jobs','writes','write_bytes','reads','read_bytes','source','ps'},'memory summary schema')
    require({k:int(v) for k,v in ends[0].items()}==dict(jobs=4,writes=3,write_bytes=192,reads=8,read_bytes=512,source=12,ps=finish),'memory summary')
    calibration=None
    for _,f in [(c,f) for c,f in extra if c=='STATE']:
        require(set(f)=={'ps','calib','ui_rst','rst'},'calibration log schema')
        if f['calib']=='1' and f['ui_rst']=='0' and calibration is None:calibration=int(f['ps'])
        if calibration is not None:require(f['calib']=='1' and f['ui_rst']=='0','runtime calibration loss')
    require(calibration is not None and calibration<int(starts[0]['ps'])<auxgroups['ARM'][0]['ps'],'cold calibration precedes traffic')
    clean=[re.sub(r'\berror=[01]\b','checked_completion_flag',s) if s.startswith('NTRPU_LINK_DONE ') else s
           for s in sim.splitlines() if not s.lstrip().startswith('#')]
    require(not any(re.search(r'\b(?:ERROR|FATAL|FAILED|MISMATCH|VIOLATION|VIOLATED|INCORRECT_PARAMETER)\b|SIMULATIONS NOT SUPPORTED|timing\s+(?:check\s+)?fail',s,re.I) for s in clean),'model/tool diagnostics')
    endings=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns)\s*:(.+)$',sim,re.M)
    require(len(endings) in {1,2} and len(set(endings))==1 and int(endings[0][0])*(1000 if endings[0][1]=='ns' else 1)==finish,'simulator terminal time')
    periodic=None
    if require_periodic:
        from check_ntrpu_ingress_periodic import review as review_periodic
        periodic=review_periodic(sim)
    return dict(status='PASS_NATIVE_FRAGMENT_HOST_DATA_COMPONENT' if require_periodic else 'TRANSFERS_REVIEWED_MODEL_WARNINGS_UNQUALIFIED',
                qualified=require_periodic,periodic=periodic,jobs=4,posted_writes=12,
                memory_writes=3,write_bytes=192,readbacks=8,readback_bytes=512,accepted_byte_writes=256,accepted_frames=8,rejected_frames=4,rejected_identity=2,
                frame_wire_payload_bytes=768,unique_staged_payload_bytes=224,
                locations=sorted(committed),native_stalls=native['stalls'],ui_stalls=aux['stalls'],
                signals=aux['signals'],final_ps=finish,calibration_ps=calibration,
                native_measurements=measurements,memory_measurements=memory_latencies,
                scope='Actual generated native PCIe/XDMA bypass, AMD asynchronous AXI, unchanged ingress/controller/receiver/capacity filter, generated cold generic MIG/model. Trusted context/isolation and UI readback are BFMs. Three locations including final128GiB line, stale frame, duplicate, STOP and recovery; paired control witness required, physical CDC latency/metastability/timing, exact DIMM, warm reset, exhaustive capacity, physical board/P2P or inference qualification.')
