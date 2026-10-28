# Data-path component only. The paired control checker must account for every
# additional RQ/RC/TAG event and AXI-Lite transfer in this 180-signal bank.
"""Sample native PCIe plus both AXI domains and compare actual MIG readback.

This is the bounded four-job fixture, not a general protocol checker or a
physical CDC proof. AMD FIFO behavioral synchronization-delay limits apply.
"""
import re
from functools import lru_cache
from check_ntrpu_pcie_bypass_native import bits, require, sample_vcd
from check_ntrpu_pcie_bypass_smoke import fields, packet_blocks
from check_ntrpu_pcie_line_link import ROOT, WIDTHS as INGRESS_WIDTHS, fixture as original_fixture
from check_ntrpu_pcie_ingress_peer_link import WIDTHS as PEER_WIDTHS

def fixture():
    return [dict(row,mask={5:3,10:12}.get(row['mask'],row['mask'])) for row in original_fixture()]

CONTROL_WIDTHS=dict(awaddr=32,wdata=32,araddr=32,wstrb=4,awvalid=1,wvalid=1,bready=1,arvalid=1,rready=1,awready=1,wready=1,arready=1,bvalid=1,rvalid=1,bresp=2,rresp=2,rdata=32)

DESTINATIONS=[0,0x100000040,0x100000040,0x1fffffffc0]
WRITE_JOBS=[0,1,3]
STATE_WIDTHS={**INGRESS_WIDTHS,**{k:v for k,v in PEER_WIDTHS.items() if not k.startswith('m_axi_')},
              'clk':1,'rst_n':1,'host_rst_n':1,'calibrated':1,'ui_rst':1,'guard_idle':1,'irq':1,'protocol_fault':1}
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
    require(set(names)==set(native['widths'])|set(extra) and len(names)==180,'exact 180-signal inventory')
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
    values={};updates={};now=0;events=[];states={};held={};stalls={c:0 for c in channels};prior_ui=None;live=False
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        nonlocal prior_ui,live
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
                require(state['calibrated']==1 and state['ui_rst']==state['protocol_fault']==0 and (state['stop']==0 or (state['epoch']==3 and state['error_code']==4)),'cold MIG state')
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
    flush();require(live and not held,'unfinished UI transfer')
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
    require([int(final[k]) for k in ['jobs','writes','lines','published_bytes']]==[4,83,3,192],'final contract')
    require([int(summary[k]) for k in ['aw','w','b','lines','results']]==[83,83,83,3,4],'logged counts')
    start,finish=int(enum['ps']),int(final['ps'])
    native,aux=sample(vcd)
    device={c:[e for e in aux['events'] if e['channel']==c] for c in ['DAW','DW','DB']}
    require(all(len(v)==83 for v in device.values()),'83 actual CDC output transfers')
    require(native['end_ps']==aux['end_ps']==finish==int(summary['ps']),'VCD final drain')
    groups={ch:[e for e in native['events'] if e['channel']==ch and e['ps']>=start and (ch!='RQ' or (9<<32)<=bits(e['tdata'],63,0)<(9<<32)+(1<<20))] for ch in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require(all(len(groups[ch])==83 for ch in ['RQ','AW','W','B']),'native write counts')
    require(all(not groups[ch] for ch in ['AR','R']),'unexpected data-window readback')
    require(all(int(summary[ch.lower()+'_stall'])==native['stalls'][ch] for ch in ['AW','W','B']),'native stalls')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start and (9<<32)<=((p['values'].get('Address High',0)<<32)|p['values'].get('Address Low',0))<(9<<32)+(1<<20)]
    rx=[p for p in packet_blocks(rx_text) if p['ps']>=start]
    require(len(tx)==83,'data-window posted request logger counts')
    require(all(len(rows(ch))==83 for ch in ['HOST_WRITE','AW','W','B']),'one log per native transfer')
    banks={j:[None]*64 for j in range(4)};bitmaps={j:0 for j in range(4)};measurements=[];accepted_bytes=0
    fixtures=fixture()
    for i,expected in enumerate(fixtures):
        job,offset,mask,word=expected['job'],expected['offset'],expected['mask'],expected['data']
        req,aw,w,b=[groups[ch][i] for ch in ['RQ','AW','W','B']]
        host=rows('HOST_WRITE')[i];packet=tx[i];v=packet['values']
        require(int(host['job'])==job and int(host['word'])==expected['word'],'host order')
        require(int(host['addr'],16)==(9<<32)+offset and int(host['data'],16)==word and int(host['mask'],16)==mask,'host stimulus')
        require(bits(req['tdata'],63,0)==(9<<32)+offset,'native RQ full address')
        require(bits(req['tdata'],74,64)==1 and bits(req['tdata'],78,75)==1,'native RQ type/length')
        require(bits(req['tdata'],79,79)==0 and bits(req['tdata'],127,120)==0,'native RQ flags')
        require(bits(req['tdata'],95,80)==0xaf and bits(req['tdata'],119,104)==0x1a0,'native request IDs')
        require(bits(req['tdata'],103,96)==int(host['tag'],16)==0x40+i,'native request tag')
        require(bits(req['tdata'],159,128)==word and bits(req['tuser'],7,0)==mask and bits(req['tuser'],15,8)==0,'native PCIe bytes')
        require(bits(req['tkeep'])==31 and bits(req['tlast'])==1,'native request framing')
        require(packet['kind']=='Memory Write-64 Frame' and packet['ps']==req['ps'],'decoded request kind/time')
        require((v['Address High']<<32)|v['Address Low']==(9<<32)+offset,'decoded full address')
        require(v['Tag']==0x40+i and v['Length']==1 and v['Last and First Byte Enables']==mask,'decoded header')
        require(packet['payload']==word.to_bytes(4,'little'),'decoded data')
        first=(mask&-mask).bit_length()-1
        require(bits(aw['addr'])==(1<<32)+offset+first,'translated first-enabled-byte address')
        require(bits(aw['id'])==0 and bits(aw['len'])==0 and bits(aw['size'])==2 and bits(aw['burst'])==1,'native AW format')
        require(bits(w['strb'])==mask<<(offset&63) and bits(w['last'])==1,'native W mask/last')
        for j,value in enumerate(word.to_bytes(4,'little')):
            if mask>>j&1:
                lane=(offset&63)+j
                require(bits(w['data'],lane*8+7,lane*8)==value,'native write byte')
                logged_data=rows('W')[i]['data']
                require(int(logged_data[len(logged_data)-2*(lane+1):len(logged_data)-2*lane],16)==value,'logged active write byte')
                if not expected['resp']:
                    require(banks[job][lane] is None or banks[job][lane]==value,'conflicting byte accepted')
                    banks[job][lane]=value;bitmaps[job]|=1<<lane;accepted_bytes+=1
        require(bits(b['id'])==0 and bits(b['resp'])==expected['resp'],'native B status')
        da,dw,db=[device[c][i] for c in ['DAW','DW','DB']]
        for k in ['addr','id','len','size','burst']:require(da[k]==bits(aw[k]),'CDC AW payload')
        require(da['lock']==0 and dw['strb']==bits(w['strb']) and dw['last']==1,'CDC attributes')
        for lane in range(64):
            if dw['strb']>>lane&1:require(bits(dw['data'],lane*8+7,lane*8)==bits(w['data'],lane*8+7,lane*8),'CDC active write bytes')
        require(db['id']==bits(b['id']) and db['resp']==bits(b['resp']),'CDC B payload')
        require(da['ps']>aw['ps'] and dw['ps']>w['ps'] and db['ps']>max(da['ps'],dw['ps']) and b['ps']>db['ps'],'CDC causal order')
        state=aux['states'][db['ps']]
        require(state['epoch']==job+1 and state['valid_bytes']==bitmaps[job],'native staged bitmap/epoch')
        require(state['failed']==int(expected['resp']!=0) and state['error_code']==(4 if expected['resp'] else 0),'native conflict state')
        for ch,n in [('AW',aw),('W',w),('B',b)]:
            logged=rows(ch)[i];require(int(logged['ps'])==n['ps'],'log/native time')
            keys={'AW':['addr','id','len','size','burst'],'W':['strb','last'],'B':['id','resp']}[ch]
            require(all(int(logged[k],16 if k in {'addr','id','strb'} else 10)==bits(n[k]) for k in keys),'log/native fields')
        require(int(host['ps'])<=req['ps']<=aw['ps']<b['ps'] and req['ps']<=w['ps']<b['ps'],'causal write chain')
        measurements.append(dict(job=job,rq_ps=req['ps'],b_ps=b['ps'],rq_to_b_ps=b['ps']-req['ps'],resp=expected['resp']))
    require(accepted_bytes==198,'accepted byte accounting')
    auxgroups={ch:[e for e in aux['events'] if e['channel']==ch and e['ps']>=start] for ch in ['ARM','LINE','DONE','RESULT']}
    require([len(auxgroups[k]) for k in ['ARM','LINE','DONE','RESULT']]==[4,3,3,4],'ingress native event counts')
    for kind in auxgroups:require(len(rows(kind))==len(auxgroups[kind]),'ingress log count '+kind)
    for kind,jobs in [('ARM',[0,1,2,3]),('LINE',[0,1,3]),('RESULT',[0,1,2,3])]:
        require([int(x['job']) for x in rows(kind)]==jobs,'logged job sequence '+kind)
    for job,offset in enumerate([0,0x80000,0x80000,0xfffc0]):
        arm,result=auxgroups['ARM'][job],auxgroups['RESULT'][job]
        astate,rstate=aux['states'][arm['ps']],aux['states'][result['ps']]
        require(arm['epoch']==result['epoch']==job+1 and arm['tag']==result['tag']==0x2000+job,'job identity')
        require(arm['window']==(1<<32)+offset and arm['destination']==DESTINATIONS[job],'job mapping')
        require(astate['source_stopped']==astate['ingress_isolated']==astate['transport_idle']==1 and astate['quiesce_epoch']==astate['epoch']==job,'arm barrier')
        require(rstate['source_stopped']==rstate['ingress_isolated']==rstate['transport_idle']==1 and rstate['quiesce_epoch']==job+1,'result barrier')
        require(result['status']==(1 if job==2 else 0),'job result')
        indices=[i for i,w in enumerate(fixtures) if w['job']==job]
        require(arm['ps']<groups['RQ'][indices[0]]['ps'] and groups['B'][indices[-1]]['ps']<result['ps'],'job transport drain')
        if job<3:require(result['ps']<auxgroups['ARM'][job+1]['ps'],'rearm after retirement')
        if job!=2:
            index=[0,1,3].index(job);line,done=auxgroups['LINE'][index],auxgroups['DONE'][index]
            require(line['epoch']==done['epoch']==job+1 and line['tag']==done['tag']==0x2000+job,'destination identity')
            require(line['address']==DESTINATIONS[job] and line['data']==int.from_bytes(bytes(banks[job]),'little'),'assembled line bytes')
            require(done['error']==0 and device['DB'][indices[-1]]['ps']<line['ps']<done['ps']<result['ps'],'destination completion chain')
        else:require(bitmaps[job]==15,'conflict retains only earlier staged bytes')
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
        logged=[f for c,f in extra if c==ch];require(len(logged)==83,'CDC log count '+ch)
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
            require(not any(arm['ps']<=e['ps']<=retired['ps'] for c in ['MAW','MW','MB'] for e in memory[c]),'conflict reached MIG')
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
    require({k:int(v) for k,v in ends[0].items()}==dict(jobs=4,writes=3,write_bytes=192,reads=8,read_bytes=512,source=83,ps=finish),'memory summary')
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
    return dict(status='PASS_NATIVE_HOST_DATA_COMPONENT' if require_periodic else 'TRANSFERS_REVIEWED_MODEL_WARNINGS_UNQUALIFIED',
                qualified=require_periodic,periodic=periodic,jobs=4,posted_writes=83,
                memory_writes=3,write_bytes=192,readbacks=8,readback_bytes=512,accepted_byte_writes=198,
                locations=sorted(committed),native_stalls=native['stalls'],ui_stalls=aux['stalls'],
                signals=aux['signals'],final_ps=finish,calibration_ps=calibration,
                native_measurements=measurements,memory_measurements=memory_latencies,
                scope='Actual generated native PCIe/XDMA bypass, AMD asynchronous AXI, unchanged ingress/controller/receiver/capacity filter, generated cold generic MIG/model. Trusted context/isolation and UI readback are BFMs. Three locations including final128GiB line, conflict and recovery; no host completion protocol, physical CDC latency/metastability/timing, exact DIMM, warm reset, exhaustive capacity, physical board/P2P or inference qualification.')
