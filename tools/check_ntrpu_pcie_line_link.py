"""Native PCIe/AXI plus ingress-control witness for the four-job link fixture."""
import re
from check_ntrpu_pcie_bypass_native import bits, require, sample_vcd
from check_ntrpu_pcie_bypass_smoke import fields, packet_blocks

ROOT = '/board/ntrpu_bypass/'
WIDTHS = {k:1 for k in ['stop','source_stopped','ingress_isolated','arm_valid','arm_ready',
                      'lease_held','failed','transport_idle','line_valid','line_ready',
                      'done_valid','done_ready','done_error','result_valid','result_ready']}
WIDTHS.update({k:64 for k in ['quiesce_epoch','arm_epoch','arm_tag','arm_window','arm_destination',
                            'epoch','valid_bytes','line_epoch','line_tag','line_address',
                            'done_epoch','done_tag','result_epoch','result_tag']})
WIDTHS.update(error_code=4,line_data=512,result_status=4)


def sample(text):
    """Keep the qualified 48-signal parser unchanged; independently sample extra ports.

    The raw VCD must contain exactly the union. Filter only named extra signals
    for the original parser; retain timestamps and original native values.
    """
    header,body=text.split('$enddefinitions $end',1)
    scope=[];names={};widths={};native_header=[];extra_codes=set()
    extra={ROOT+k:v for k,v in WIDTHS.items()}
    extra['/board/EP/m_axib_awlock']=1
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+ '/'.join(scope+[f[4]])
            require(name not in names,'duplicate raw VCD name')
            names[name]=f[3];widths[f[3]]=int(f[2])
            if name in extra:
                require(int(f[2])==extra[name],'extra signal width '+name)
                keep=False;extra_codes.add(f[3])
        if keep:native_header.append(line)
    require(set(extra).issubset(names),'missing ingress signals')
    native_codes={code for name,code in names.items() if name not in extra}
    def decode(line):
        if line.startswith('b'):return line[1:].split()
        return line[0],line[1:]
    filtered=[]
    for line in body.splitlines():
        clean=line.strip()
        if not clean or clean[0] in '$#':filtered.append(line)
        elif decode(clean)[1] in native_codes:filtered.append(line)
    native=sample_vcd('\n'.join(native_header)+'\n$enddefinitions $end\n'+'\n'.join(filtered))
    require(set(names)==set(native['widths'])|set(extra),'exact combined VCD inventory')
    values={};updates={};now=0;events=[];states={};held={};stalls={c:0 for c in ['ARM','LINE','DONE','RESULT']}
    channels={'ARM':('arm_', ['epoch','tag','window','destination']),
              'LINE':('line_', ['epoch','tag','address','data']),
              'DONE':('done_', ['epoch','tag','error']),
              'RESULT':('result_', ['epoch','tag','status'])}
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        clock=names['/board/EP/user_clk']
        if values.get(clock)=='0' and updates.get(clock)=='1' and raw('/board/EP/user_resetn')=='1':
            state={k:bits(raw(ROOT+k)) for k in WIDTHS}
            require(state['stop']==0,'unexpected stop in link fixture')
            if raw('/board/EP/m_axib_awvalid')=='1':
                require(bits(raw('/board/EP/m_axib_awlock'))==0,'native AWLOCK')
            states[now]=state
            for ch,(prefix,fs) in channels.items():
                payload={f:state[prefix+f] for f in fs}
                valid,ready=state[prefix+'valid'],state[prefix+'ready']
                if ch in held:
                    require(valid==1 and payload==held.pop(ch),'ingress '+ch+' changed while stalled')
                if valid:
                    if ready:events.append(dict(channel=ch,ps=now,**payload))
                    else:held[ch]=payload;stalls[ch]+=1
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            flush();updates={};new=int(line[1:]);require(new>=now,'raw VCD time reversal');now=new
        else:
            raw_value,code=decode(line);raw_value=raw_value.lower()
            require(code in widths and set(raw_value)<=set('01xz'),'bad raw VCD value')
            require(len(raw_value)<=widths[code],'raw VCD overflow')
            updates[code]=raw_value.rjust(widths[code],raw_value[0] if raw_value[0] in 'xz' else '0')
    flush();require(not held,'undrained ingress handshake')
    return native,dict(events=events,states=states,stalls=stalls,end_ps=now,signals=len(names))


def fixture():
    writes=[]
    for job,offset in enumerate([0,0x80000,0x80000,0xfffc0]):
        pairs=[(0,15,False),(0,15,True)] if job==2 else []
        if job!=2:
            for word in range(16):
                if job==0:pairs.append((word,15,False))
                else:
                    pairs.append((word,5 if job==1 else 1,False))
                    if job==1 and word==0:pairs.append((word,5,False))
                    pairs.append((word,10 if job==1 else 14,False))
        for word,mask,conflict in pairs:
            data=int.from_bytes(bytes((job*53+(word*4+j)*29+23)&255 for j in range(4)),'little')
            if conflict:data^=1
            writes.append(dict(job=job,offset=offset+word*4,word=word,mask=mask,data=data,resp=2 if conflict else 0))
    require(len(writes)==83,'fixture size')
    return writes


def review(sim,tx_text,rx_text,vcd):
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',sim,re.M|re.I),'simulation failure')
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
    require(native['end_ps']==aux['end_ps']==finish==int(summary['ps']),'VCD final drain')
    groups={ch:[e for e in native['events'] if e['channel']==ch and e['ps']>=start] for ch in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require(all(len(groups[ch])==83 for ch in ['RQ','AW','W','B']),'native write counts')
    require(all(not groups[ch] for ch in ['RC','TAG','AR','R']),'unexpected native reads/completions')
    require(all(int(summary[ch.lower()+'_stall'])==native['stalls'][ch] for ch in ['AW','W','B']),'native stalls')
    tx=[p for p in packet_blocks(tx_text) if p['ps']>=start]
    rx=[p for p in packet_blocks(rx_text) if p['ps']>=start]
    require(len(tx)==83 and not rx,'posted request logger counts')
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
        state=aux['states'][b['ps']]
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
        require(arm['window']==(1<<32)+offset and arm['destination']==0x1fffffffc0,'job mapping')
        require(astate['source_stopped']==astate['ingress_isolated']==astate['transport_idle']==1 and astate['quiesce_epoch']==astate['epoch']==job,'arm barrier')
        require(rstate['source_stopped']==rstate['ingress_isolated']==rstate['transport_idle']==1 and rstate['quiesce_epoch']==job+1,'result barrier')
        require(result['status']==(4 if job==2 else 0),'job result')
        indices=[i for i,w in enumerate(fixtures) if w['job']==job]
        require(arm['ps']<groups['RQ'][indices[0]]['ps'] and groups['B'][indices[-1]]['ps']<result['ps'],'job transport drain')
        if job<3:require(result['ps']<auxgroups['ARM'][job+1]['ps'],'rearm after retirement')
        if job!=2:
            index=[0,1,3].index(job);line,done=auxgroups['LINE'][index],auxgroups['DONE'][index]
            require(line['epoch']==done['epoch']==job+1 and line['tag']==done['tag']==0x2000+job,'destination identity')
            require(line['address']==0x1fffffffc0 and line['data']==int.from_bytes(bytes(banks[job]),'little'),'assembled line bytes')
            require(done['error']==0 and groups['B'][indices[-1]]['ps']<line['ps']<done['ps']<result['ps'],'destination completion chain')
        else:require(bitmaps[job]==15,'conflict retains only earlier staged bytes')
    for kind,mapping in [('ARM',{'window':'window','dest':'destination','epoch':'epoch','tag':'tag'}),
                         ('LINE',{'addr':'address','data':'data','epoch':'epoch','tag':'tag'}),
                         ('DONE',{'epoch':'epoch','tag':'tag','error':'error'}),
                         ('RESULT',{'epoch':'epoch','tag':'tag','status':'status'})]:
        for logged,n in zip(rows(kind),auxgroups[kind]):
            require(int(logged['ps'])==n['ps'],'egress log time')
            require(all(int(logged[k],10 if k in {'status','error'} else 16)==n[v] for k,v in mapping.items()),'egress log fields')
    endings=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns)\s*:(.+)$',sim,re.M)
    require(len(endings) in {1,2} and len(set(endings))==1 and int(endings[0][0])*(1000 if endings[0][1]=='ns' else 1)==finish,'simulator terminal time')
    return dict(status='PASS_NATIVE_PCIE_LINE_INGRESS',qualified=True,jobs=4,writes=83,published_lines=3,
                accepted_byte_writes=198,published_bytes=192,conflicts=1,native_stalls=native['stalls'],
                ingress_stalls=aux['stalls'],signals=aux['signals'],final_ps=finish,measurements=measurements,
                scope='Actual generated XDMA bypass into unchanged qualified ingress RTL. Test-only destination and isolation controls; no host-visible completion protocol, receiver/MIG/P2P/routed/physical qualification.')
