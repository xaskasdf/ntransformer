"""Reconstruct sampled native PCIe/AXI transfers from an XSIM VCD.

Descriptors follow AMD PG213. Sample values before each rising edge, before
same-timestamp NBA updates; retain four-state payloads and reject unknown
active bits. This bounded witness is not a general PCIe packet decoder.
"""
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(value, high=None, low=0):
    if high is None:
        high=len(value)-1
    selected=value[len(value)-1-high:len(value)-low]
    require(bool(selected) and set(selected)<=set('01'),'unknown active bits')
    return int(selected,2)


def sample_vcd(text):
    header,body=text.split('$enddefinitions $end',1)
    require(re.search(r'\$timescale\s+1ps\s+\$end',header),'expected 1 ps VCD timescale')
    scope=[]; names={}; widths={}
    for line in header.splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='$scope':scope.append(fields[2])
        elif fields[0]=='$upscope':scope.pop()
        elif fields[0]=='$var':
            name='/'+ '/'.join(scope+[fields[4]])
            require(name not in names,'duplicate VCD variable')
            names[name]=fields[3];widths[fields[3]]=int(fields[2])
    channels={
        'RQ':('/board/RP/','s_axis_rq_',('tdata','tkeep','tuser','tlast')),
        'RC':('/board/RP/','m_axis_rc_',('tdata','tkeep','tuser','tlast')),
    }
    for ch,fs in {'AW':['addr','id','len','size','burst'],'W':['data','strb','last'],
                  'B':['id','resp'],'AR':['addr','id','len','size','burst'],'R':['id','data','resp','last']}.items():
        channels[ch]=('/board/EP/','m_axib_'+ch.lower(),tuple(fs))
    required={'/board/EP/user_resetn':1,'/board/EP/user_clk':1,'/board/RP/user_clk':1}
    for ch,(root,prefix,fs) in channels.items():
        for suffix in fs+('tvalid','tready') if ch in {'RQ','RC'} else fs+('valid','ready'):
            required[root+prefix+suffix]=None
    for i in range(2):
        required['/board/RP/pcie_4_0_rport/pcie_rq_tag'+str(i)]=8
        required['/board/RP/pcie_4_0_rport/pcie_rq_tag_vld'+str(i)]=1
    require(set(names)==set(required),'exact native signal inventory')
    for n,w in required.items():
        if w:require(widths[names[n]]==w,'native signal width '+n)
    for ch,(root,prefix,fs) in channels.items():
        expected={'addr':64,'id':4,'len':8,'size':3,'burst':2,'data':512,'strb':64,'last':1,'resp':2,
                  'tdata':512,'tkeep':16,'tlast':1,'tuser':137 if ch=='RQ' else 161}
        for suffix in fs:
            require(widths[names[root+prefix+suffix]]==expected[suffix],'exact native bus width '+ch+suffix)
    values={};updates={}; now=0;events=[];held={};stalls={ch:0 for ch in channels}
    def value(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        for root in ['/board/RP/','/board/EP/']:
            clock=names[root+'user_clk']
            if values.get(clock)!='0' or updates.get(clock)!='1':continue
            if root=='/board/EP/' and value(root+'user_resetn')!='1':continue
            for ch,(domain,prefix,fs) in channels.items():
                if domain!=root:continue
                valid=value(root+prefix+('tvalid' if ch in {'RQ','RC'} else 'valid'))
                ready=value(root+prefix+('tready' if ch in {'RQ','RC'} else 'ready'))
                payload={f:value(root+prefix+f) for f in fs}
                if ch in held:
                    require(valid=='1' and payload==held[ch],ch+' changed while stalled at '+str(now))
                    del held[ch]
                if valid=='1':
                    require(ready in {'0','1'},'unknown ready on active '+ch)
                    if ready=='1':events.append(dict(channel=ch,ps=now,**payload))
                    else:held[ch]=payload;stalls[ch]+=1
            if root=='/board/RP/':
                for i in range(2):
                    prefix=root+'pcie_4_0_rport/pcie_rq_tag'
                    if value(prefix+'_vld'+str(i))=='1':
                        events.append(dict(channel='TAG',ps=now,slot=i,tag=bits(value(prefix+str(i)))))
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            flush();updates={};new=int(line[1:]);require(new>=now,'VCD time reversal');now=new
        else:
            if line.startswith('b'):raw,ident=line[1:].split()
            else:raw,ident=line[0],line[1:]
            require(ident in widths and set(raw.lower())<=set('01xz'),'invalid VCD value')
            raw=raw.lower();pad=raw[0] if raw[0] in 'xz' else '0'
            require(len(raw)<=widths[ident],'overwide VCD value')
            updates[ident]=raw.rjust(widths[ident],pad)
    flush()
    require(not held,'unfinished stalled transfer')
    return dict(events=events,stalls=stalls,end_ps=now,widths={n:widths[i] for n,i in names.items()})


def review_native(sampled, start_ps, final_ps):
    events=[e for e in sampled['events'] if e['ps']>=start_ps]
    groups={ch:[e for e in events if e['channel']==ch] for ch in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require(sampled['end_ps']==final_ps,'VCD final timestamp')
    require(len(groups['RQ'])==6,'six native requests after enumeration')
    require(all(len(groups[ch])==3 for ch in ['RC','TAG','AW','W','B','AR','R']),'three native transfers per channel')
    require(events==sorted(events,key=lambda e:e['ps']) and all(e['ps']<=final_ps for e in events),'native event timeline')
    rows=[]
    for i,(offset,word) in enumerate(zip([0,0x80000,0xfffc0],[0x10234567,0x89abcdef,0xfedcba98])):
        write,read=groups['RQ'][i],groups['RQ'][i+3]
        rc,tag,aw,w,b,ar,r=[groups[ch][i] for ch in ['RC','TAG','AW','W','B','AR','R']]
        for req,kind in [(write,1),(read,0)]:
            data=req['tdata'];user=req['tuser']
            require(bits(data,63,0)==(9<<32)+offset,'native RQ address/type')
            require(bits(data,74,64)==1 and bits(data,78,75)==kind,'native RQ size/type')
            require(bits(data,79,79)==0 and bits(data,126,121)==0,'native RQ poison/attributes')
            # The installed public TX example drives tag_[9]/tag_[8] into
            # bits 127/120 when its 10-bit tag parameter is TRUE. Record this
            # exact stimulus separately; it does not prove wire ECRC behavior.
            require(bits(data,127,127)==1 and bits(data,120,120)==1,'public TX high tag stimulus bits')
            require(bits(data,95,80)==0xaf and bits(data,119,104)==0x1a0,'native request IDs')
            require(bits(user,7,0)==15 and bits(user,15,8)==0,'native RQ byte enables')
            require(bits(req['tkeep'])==(31 if kind else 15) and bits(req['tlast'])==1,'native RQ keep/last')
        require(bits(write['tdata'],159,128)==word,'native RQ write payload')
        for a in [aw,ar]:
            require(bits(a['addr'])==(1<<32)+offset,'native AXI translated address')
            require(bits(a['id'])==0 and bits(a['len'])==0 and bits(a['size'])==2 and bits(a['burst'])==1,'native AXI request attributes')
        require(bits(w['data'],31,0)==word and bits(w['strb'])==15 and bits(w['last'])==1,'native AXI W payload/strobes')
        require(bits(b['id'])==bits(aw['id']) and bits(b['resp'])==0,'native AXI B identity/status')
        require(bits(r['id'])==bits(ar['id']) and bits(r['resp'])==0 and bits(r['last'])==1,'native AXI R identity/status')
        require(bits(r['data'])==int.from_bytes(b'\xa5'*60+word.to_bytes(4,'big'),'big'),'native AXI R payload/sentinels')
        d=rc['tdata']
        require(bits(d,11,0)==offset%4096 and bits(d,15,12)==0,'native RC lower address/error')
        require(bits(d,28,16)==4 and bits(d,29,29)==0 and bits(d,30,30)==1,'native RC byte count/completed')
        require(bits(d,42,32)==1 and bits(d,46,43)==0,'native RC length/status/poison')
        require(bits(d,71,64)==tag['tag'],'allocated native tag matches completion')
        require(bits(d,63,48)==0xaf and bits(d,87,72)==0x1a0 and bits(d,94,89)==0,'native RC IDs/attributes')
        require(bits(d,127,96)==word,'native RC payload')
        require(bits(rc['tkeep'])==15 and bits(rc['tlast'])==1,'native RC keep/last')
        require(bits(rc['tuser'],63,0)==0xf000 and bits(rc['tuser'],67,64)==1 and bits(rc['tuser'],96,96)==0,'native RC byte enables/SOP/discontinue')
        require(write['ps']<=aw['ps']<b['ps'] and write['ps']<=w['ps']<b['ps'],'native write causal order')
        require(read['ps']<=tag['ps']<rc['ps'] and read['ps']<=ar['ps']<r['ps']<=rc['ps'],'native read causal order')
        rows.append(dict(offset=offset,requested_tag=bits(read['tdata'],103,96),allocated_tag=tag['tag'],
            completion_tag=bits(d,71,64),read_request_ps=read['ps'],tag_ps=tag['ps'],completion_ps=rc['ps'],
            byte_count=bits(d,28,16),word=word))
    require(max(e['ps'] for e in groups['B'])<groups['RQ'][3]['ps'],'write drain before reads')
    return dict(status='PASS_NATIVE_BYPASS_DWORD_WITNESS',rows=rows,stalls=sampled['stalls'],end_ps=sampled['end_ps'])
