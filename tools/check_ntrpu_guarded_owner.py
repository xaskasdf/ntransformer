"""Independent NTRD ownership/MMIO and reset-causality oracle for the guarded host.

Uses explicit fixture frames and port handshakes, not phase decoding, to count
ownership. Memory completion/peer contracts are controlled, not MIG/PCIe claims.
"""
import re
from collections import Counter
from check_ntrpu_pcie_fragment_pool import FIELDS,fields,require,MAX
from check_ntrpu_pcie_fragment_dual_managed import payload
CASES=list(range(9))
LINE_CASES={0,2,3,4,5,7,8}
DONE={0:0,5:0}
RESULT={0:0,5:0}
FRAMES={j:([(0,0,0),(1,0,0)] if j in LINE_CASES else [(0,0,0)] if j==1 else []) for j in CASES}
CONTROL_FIELDS='cycle aw av ar wd ws wv wr bv br bs ra rv rr ov ore od os irq fault'.split()

def review_control(text,seed,case_id):
    raw_own=[fields(x) for x in text.splitlines() if x.startswith('OWN_TRACE ')]
    rows=[fields(x) for x in text.splitlines() if x.startswith('CTL_TRACE ')]
    require(len(rows)==len(raw_own) and [int(x['cycle']) for x in rows]==list(range(1,len(rows)+1)), 'complete control cycle stream')
    writes=[];reads=[];blocked=Counter();stats=Counter();snapshots=[]
    regs={};arm=[0]*4;channel=0;queued=running=stopped=fault=False;accepted=retired=0
    aw=wd=None;bv=rv=False;bresp=rresp=rdata=0;write_receipt=read_address=None
    reset_intervals=0;prior_reset=False
    for raw,raw_s in zip(rows,raw_own):
        require(set(raw)==set(CONTROL_FIELDS), 'control cycle schema')
        reset=raw_s['rst']=='0'
        if reset:
            require(all(raw[k]=='0' for k in ('ar','wr','rr')),'AXI-Lite reset admission')
            if not prior_reset:reset_intervals+=1
            prior_reset=True
            regs={k:0 for k in [0x14,*range(0x20,0x50,4),0x90,0x94,0x98,0xac,0xb0]}
            arm=[0]*4;channel=0;queued=running=stopped=fault=False;accepted=retired=0
            aw=wd=None;bv=rv=False;bresp=rresp=rdata=0;write_receipt=read_address=None
            continue
        prior_reset=False
        require(all(re.fullmatch('[0-9a-f]+',v) for v in raw.values()),'known active control ports')
        c={k:int(v,10 if k=='cycle' else 16) for k,v in raw.items()}
        s={k:int(v,10 if k in ('cycle','case') else 16) for k,v in raw_s.items()}
        unavailable=bool(s['invalid'] or s['fault'] or s['exhausted'] or s['pf'])
        irq=bool(regs[0x14] and (s['rv'] or fault or unavailable))
        require((c['irq'],c['fault'])==(irq,fault),'control IRQ and sticky protocol fault')
        require((s['av'],s['stop'],s['ae'],s['at'],s['aw'],s['aa'],s['ac'])==(queued,stopped,*arm,channel),'captured immutable descriptor/channel and queued ARM')
        ack=[regs[0x40]|regs[0x44]<<32,regs[0x48]|regs[0x4c]<<32]
        require((s['ke'],s['kt'])==tuple(ack),'ACK words full width')
        require((c['ar'],c['wr'],c['rr'])==(aw is None and not bv,wd is None and not bv,not rv),'AXI-Lite independent request readiness')
        require((c['bv'],c['bs'],c['ov'],c['os'],c['od'])==(bv,bresp,rv,rresp,rdata),'registered B/R and held payloads')
        if bv and not c['br']:stats['b_stall']+=1
        if rv and not c['ore']:stats['r_stall']+=1
        if queued and not s['ar']:stats['arm_stall']+=1
        def valid_address(a):return a%4==0 and 0<=a<=0xc0
        def read_register(a):
            if a in regs:return regs[a]
            table={0:0x4452544e,4:0x10000,8:0xff,0x10:0,0x18:accepted,0x1c:retired,
                   0x50:s['epoch']&0xffffffff,0x54:s['epoch']>>32,0x58:s['bytes']&0xffffffff,0x5c:s['bytes']>>32,
                   0x60:s['re']&0xffffffff,0x64:s['re']>>32,0x68:s['rt']&0xffffffff,0x6c:s['rt']>>32,
                   0x70:s['rs'],0x74:s['error'],0x78:s['stopped']|s['isolated']<<1,0x7c:s['qe']&0xffffffff,0x80:s['qe']>>32,
                   0x84:arm[0]&0xffffffff,0x88:arm[0]>>32,0x8c:int(fault),0x9c:0x5052544e,0xa0:0x31304655,0xa4:64,0xa8:32,
                   0xb4:channel,0xb8:s['invalid']|s['bh']<<1|s['kr']<<2|s['admit']<<3|s['exhausted']<<4|s['fault']<<5|s['pf']<<6,
                   0xbc:2,0xc0:37}
            table[0xc]=(int(irq)|int(queued)<<1|int(running)<<2|s['rv']<<3|int(stopped)<<4|int(fault)<<5|
                        s['ar']<<6|s['held']<<7|s['fail']<<8|s['ti']<<9|s['idle']<<10|s['bh']<<11)
            require(a in table,'defined profile register');return table[a]
        # Sample reads before any concurrent writes alter the register file.
        new_read=None
        if c['rv'] and c['rr']:
            a=c['ra'];new_read=(a,0 if valid_address(a) else 3,read_register(a) if valid_address(a) else 0)
        if c['bv'] and c['br']:
            require(write_receipt is not None,'B is backed by one write')
            writes.append((*write_receipt,bresp));write_receipt=None;bv=False
        if c['ov'] and c['ore']:
            require(read_address is not None,'R is backed by one read')
            reads.append((read_address,rdata,rresp));read_address=None;rv=False
        commit=aw is not None and wd is not None and not c['bv']
        response=0;command=0;mask=0;new_desc=None
        unexpected=bool(s['rv'] and (not running or (s['re'],s['rt'])!=tuple(arm[:2])))
        if commit:
            value,strobe=wd;mask=sum(0xff<<(8*i) for i in range(4) if strobe>>i&1);command=value&mask
            if not valid_address(aw):response=3
            elif aw==0x10:
                descriptor=[regs[a]|regs[a+4]<<32 for a in range(0x20,0x40,8)]
                good=descriptor[0]>s['epoch'] and descriptor[2]%64==0 and descriptor[3]%64==0 and descriptor[3]<1<<37 and regs[0xb0]<=1
                if command in (0,8):pass
                elif command==1:
                    response=2 if queued or running or s['rv'] or fault or unexpected or unavailable or not good else 0
                    new_desc=descriptor
                    if response and unavailable:blocked['unavailable_arm']+=1
                elif command==2:response=2 if not running or stopped else 0
                elif command==4:
                    identity=(s['re'],s['rt'])==tuple(arm[:2])==tuple(ack)
                    response=2 if not running or not s['rv'] or not identity or fault or s['invalid'] or not s['kr'] else 0
                    if response:
                        if not identity:blocked['identity']+=1
                        if not s['stopped'] or not s['isolated'] or s['qe']!=s['epoch']:blocked['barrier']+=1
                        if not s['ti'] or s['awv'] or s['wv']:blocked['transport']+=1
                        if not s['drained']:blocked['drain']+=1
                        if s['invalid']:blocked['session']+=1
                else:response=2
            elif aw==0x14:response=2 if command & 0xfffffffe else 0
            elif aw==0xb0:
                merged=(regs[aw]&~mask)|command
                response=2 if queued or running or s['rv'] or fault or unavailable or merged>1 else 0
                if response:
                    if merged>1:blocked['channel_range']+=1
                    if queued:blocked['channel_queued']+=1
                    if running:blocked['channel_running']+=1
            elif 0x20<=aw<=0x3c:response=2 if queued or running or s['rv'] or fault or unavailable else 0
            elif not 0x40<=aw<=0x4c:response=2
        require(s['kv']==bool(commit and aw==0x10 and command==4 and response==0),'ACK pulse iff MMIO retirement committed')
        if unexpected:fault=True
        if s['av'] and s['ar']:queued=False;running=True;accepted=(accepted+1)&0xffffffff;stats['accepted']+=1
        if new_read:
            read_address,rresp,rdata=new_read;rv=True
        if commit:
            write_receipt=(aw,*wd);bresp=response;bv=True
            if response==0:
                if aw==0x10:
                    if command==1:arm=new_desc;channel=regs[0xb0];queued=True;stopped=False
                    elif command==2:stopped=True
                    elif command==4:running=False;retired=(retired+1)&0xffffffff;stats['retired']+=1
                    elif command==8:
                        regs[0x90]=s['frames'];regs[0x94]=s['identity'];regs[0x98]=s['faults']<<4|s['code'];regs[0xac]=1
                        snapshots.append((regs[0x90],regs[0x94],regs[0x98]))
                elif aw==0x14:
                    if wd[1]&1:regs[aw]=wd[0]&1
                else:regs[aw]=(regs[aw]&~mask)|command
            aw=wd=None
        if c['av'] and c['ar']:aw=c['aw']
        if c['wv'] and c['wr']:wd=(c['wd'],c['ws'])
    require(not bv and not rv and wd is None and write_receipt is None and read_address is None,'control response inventory drained')
    require(aw==(0x10 if case_id==8 else None),'only partial control AW fixture remains')
    require(not queued and running==bool(case_id) and reset_intervals==1,'retained MMIO running state')
    require(stats['accepted']==1 and stats['retired']==int(case_id==0),'one ARM and expected retirement')
    require(reads[:3]==[(0,0x4452544e,0),(4,0x10000,0),(8,0xff,0)],'NTRD discovery profile')
    return dict(writes=writes,reads=reads,stats=dict(stats),pending_aw=aw)

def review_owner(text,case_id):
    seed=1
    rows=[fields(x) for x in text.splitlines() if x.startswith('OWN_TRACE ')]
    require(rows and [int(x['cycle']) for x in rows]==list(range(1,len(rows)+1)),'complete source cycle trace')
    counts=Counter({k:0 for k in ['arms','lines','commands','data','done','acks','frames','rejects','completions','publications']})
    bycase={j:Counter() for j in CASES};stalls=Counter();metrics=[];blocked=Counter()
    ctx=None;epoch=0;active=False;seen=False;lost=False;cookie=0xffffffff
    hold={};aw=None;wd=None;pending=None;frame_index=Counter();prior_b=False
    rejects=0;identity=0;resets=0;reset_seen=False;observed_cases=[];previous=None
    final=None
    for raw in rows:
        require(set(raw)==set(FIELDS),'trace schema')
        cycle=int(raw['cycle']);job=int(raw['case'])
        if raw['rst']=='0':
            if not reset_seen:resets+=1
            reset_seen=True
            require(all(raw[k]=='0' for k in ['ar','kr','rc','awr','wr','cv','dv','ov']), 'cold reset quiescence')
            ctx=None;epoch=0;active=False;seen=False;lost=False;cookie=0xffffffff
            hold={};aw=None;wd=None;pending=None;prior_b=False;rejects=identity=0;previous=None
            continue
        reset_seen=False
        require(all(re.fullmatch('[0-9a-f]+',v) for v in raw.values()),'known active ports')
        s={k:int(v,10 if k in ['cycle','case'] else 16) for k,v in raw.items()}
        if job and job not in observed_cases:observed_cases.append(job)
        require(s['epoch']==s['be']==epoch, 'joint retained epoch')
        require(s['held']==bool(ctx) and s['bh']==active, 'collector retained context versus active pool lease')
        require(s['next']==cookie and not s['exhausted'], 'full64 cookie allocation')
        invalid=lost or s['restart'] or (seen and not s['admit'])
        require(s['lost']==lost and s['invalid']==bool(invalid), 'sticky session lifetime')
        require(s['fault']==s['rejected']==0, 'no synthetic protocol fault')
        require(s['err']==(s['os'] not in (0,1)), 'explicit DONE status translation')
        require(s['pr']==(0 if invalid else 3), 'completion drain policy')
        require(s['ar']==(s['car'] and s['bar']), 'joint ready')
        require(s['rc']==(s['kv'] and s['kr']), 'single atomic result consumption')
        barrier=s['stopped'] and s['isolated'] and s['qe']==epoch
        transport=s['ti'] and not s['awv'] and not s['wv']
        ack_identity=s['ke']==s['re'] and s['kt']==s['rt']
        expected_ready=s['rv'] and ack_identity and barrier and transport and s['rr'] and not invalid
        require(s['kr']==bool(expected_ready), 'ACK full identity/live/drain/barrier predicate')
        if s['kv'] and not s['kr']:
            for reason,value in [('identity',ack_identity),('barrier',barrier),('transport',transport),('retirement',s['rr']),('session',not invalid)]:
                if not value:blocked[reason]+=1
        channels={'B':('bv','br',['bi','bs']), 'line':('lv','lr',['le','lt','la','ld']),
                  'command':('cv','cr',['cc','ca','ct','cl']), 'data':('dv','dr',['data']),
                  'DONE':('ov','ore',['oe','ot','os','ops']), 'result':('rv','rc',['re','rt','rs'])}
        for name,(v,r,p) in channels.items():
            value=tuple(s[k] for k in p)
            if name in hold:require(s[v] and hold.pop(name)==value,'stable held '+name)
            if s[v] and not s[r]:hold[name]=value;stalls[name]+=1
        def event(name):counts[name]+=1;bycase[job][name]+=1
        if s['ar']:
            require(not active and not s['stop'] and not invalid and s['admit'] and s['drained'] and barrier and transport and not s['rv'], 'ARM admission proof')
            require(s['ae']>epoch and s['aa']<1<<37 and s['aa']%64==s['aw']%64==0,'descriptor range/epoch')
        if s['av'] and s['ar']:
            require(job in CASES and bycase[job]['arms']==0, 'one ARM per fixture case')
            require(s['ae']==(epoch+1 if epoch else 0x100000001) and s['at']==0x800000010000abcd,'full64 fixture identity')
            require(s['ac']==job%2 and s['aa']==(0x1fffffffc0 if job%2 else 0x100000040) and s['aw']==0x1000fffc0,'explicit channel and high local address')
            ctx=dict(job=job,epoch=s['ae'],tag=s['at'],address=s['aa'],channel=s['ac'],
                     line=None,command=None,data=None,cpl=None,done=None,result=None,acked=False,cookie=None)
            active=True;epoch=s['ae'];event('arms')
        if s['lv']:
            require(active and ctx and job in LINE_CASES and ctx['line'] is None,'one LINE under lease')
            require((s['le'],s['lt'],s['la'],s['ld'])==(ctx['epoch'],ctx['tag'],ctx['address'],payload(job,seed)), 'assembled payload/identity')
            require(s['bytes']==MAX, 'complete byte coverage before LINE')
            if s['lr']:ctx['line']=cycle;ctx['cookie']=cookie;cookie+=1;event('lines')
        if s['cv']:
            require(active and ctx and ctx['line'] is not None and ctx['command'] is None,'unique command after LINE')
            require((s['cc'],s['ca'],s['ct'],s['cl'])==(ctx['channel'],ctx['address'],ctx['cookie'],0),'registered channel/address/cookie')
            if s['cr']:ctx['command']=cycle;event('commands')
        if s['dv']:
            require(active and ctx and ctx['command'] is not None and ctx['command']<cycle and ctx['data'] is None,'unique data after command')
            require(s['data']==payload(job,seed),'all 64 data bytes')
            if s['dr']:ctx['data']=cycle;event('data')
        takes=s['pv']&s['pr']
        if takes:
            require(active and ctx and ctx['data'] is not None and ctx['data']<cycle and ctx['cpl'] is None,'owned response after data')
            require(takes==1<<ctx['channel'] and s['pt']==ctx['cookie']<<(64*ctx['channel']), 'full cookie/channel completion')
            expected=0
            require(s['ps']==expected<<(2*ctx['channel']), 'fixture completion status')
            ctx['cpl']=cycle;event('completions')
        if s['ov']:
            require(active and ctx and ctx['cpl'] is not None and ctx['cpl']<cycle and ctx['done'] is None, 'DONE requires actual completion')
            require((s['oe'],s['ot'],s['os'],s['ops'])==(ctx['epoch'],ctx['tag'],DONE[job],0),'DONE identity/status')
            require(not invalid and s['drained'], 'drained live DONE in this fixture')
            if s['ore']:ctx['done']=cycle;event('done')
        if s['rv']:
            require(ctx and (active or ctx['acked']), 'retained result context')
            require((s['re'],s['rt'],s['rs'])==(ctx['epoch'],ctx['tag'],RESULT[job]),'visible RESULT full identity/status')
            if ctx['result'] is None:
                require(barrier and transport, 'RESULT external epoch barrier and drained AXI')
                require(ctx['done'] is not None , 'no result before destination completion')
                ctx['result']=cycle;event('publications')
        if s['kr']:
            require(active and ctx and ctx['result'] is not None and not invalid and s['drained'] and not s['pv'], 'ACK cannot discard active/invalid destination')
            require(ctx['done'] is not None , 'ACK completion or canceled partial line')
        if s['rc']:
            require(active and ctx and not ctx['acked'] and s['ke']==ctx['epoch'] and s['kt']==ctx['tag'],'one ACK for full context')
            ctx['acked']=True;active=False;event('acks')
            metrics.append(dict(job=job,line_to_ack_cycles=cycle-ctx['line'] if ctx['line'] else None,
                                command_to_ack_cycles=cycle-ctx['command'] if ctx['command'] else None))
        # Independently pair AXI AW/W and check the complete 64-byte frame.
        if s['awv'] and s['awr']:
            require(aw is None and pending is None,'one outstanding AW')
            require((s['awa'],s['awi'],s['awl'],s['aws'],s['awb'],s['awk'])==(0x1000fffc0,job%16,0,6,1,0),'frame AW')
            aw=job
        if s['wv'] and s['wr']:
            require(wd is None and pending is None,'one outstanding W')
            require(s['ws']==MAX and s['wl']==1,'full framed W')
            wd=s['wd']
        if aw is not None and wd is not None:
            require(aw==job and ctx is not None,'AW/W lease pairing')
            idx=frame_index[job];require(idx<len(FRAMES[job]),'no extra frame')
            half,variant,resp=FRAMES[job][idx]
            data=payload(job,seed)^(1 if variant==1 else 0)
            e=ctx['epoch']^((1<<32) if variant==2 else 0)
            expected=0x313046555052544e | e<<64 | ctx['tag']<<128 | (half*32)<<192 | ((data>>(half*256))&((1<<256)-1))<<256
            require(wd==expected,'exact F01 header and all payload bytes')
            pending=(resp,variant,job);frame_index[job]+=1;aw=wd=None
        if s['bv']:
            require(pending is not None and (s['bi'],s['bs'])==(pending[2]%16,pending[0]),'B matches staged frame')
            if not prior_b:
                if pending[0]:rejects+=1
                if pending[1]==2:identity+=1
            if s['br']:
                event('frames')
                if pending[0]:event('rejects')
                pending=None
        require((s['frames'],s['identity'])==(rejects,identity),'frame reject ledger')
        prior_b=bool(s['bv'] and not s['br'])
        seen=seen or bool(s['admit']);lost=bool(invalid);previous=s;final=s
    require(resets==1,'one cold lifetime only')
    require(frame_index==Counter({case_id:len(FRAMES[case_id])}) or (case_id==6 and not frame_index),'complete fixture frames')
    expected=dict(arms=1,lines=int(case_id in LINE_CASES),commands=int(case_id in LINE_CASES),data=int(case_id in LINE_CASES),done=int(case_id in DONE),acks=int(case_id==0),frames=len(FRAMES[case_id]),rejects=0,completions=int(case_id in DONE),publications=int(case_id in RESULT))
    require(dict(counts)==expected,'complete ownership event inventory '+str(dict(counts)))
    require(final['invalid']==bool(case_id) and final['bh']==bool(case_id) and final['held']==1,'retained ownership at end')
    require(aw==(6 if case_id==6 else None) and wd is None and pending is None,'only partial data AW remains')
    end=[fields(x) for x in text.splitlines() if x.startswith('OWNER_PASS ')]
    require(len(end)==1 and int(end[0]['case'])==case_id,'one terminal case')
    for key in ['arms','lines','commands','data','done','acks','frames']:require(int(end[0][key])==counts[key],'terminal count '+key)
    for key,port in [('held','bh'),('invalid','invalid'),('epoch','epoch'),('next','next')]:require(int(end[0][key],16)==final[port],'terminal retained '+key)
    return dict(case=case_id,source_samples=len(rows),counts=dict(counts),stalls=dict(stalls),metrics=metrics)

def decoded(text,prefix):
    out=[]
    for line in text.splitlines():
        if not line.startswith(prefix+' '):continue
        raw=fields(line)
        require(all(re.fullmatch('[0-9a-f]+',x) for x in raw.values()),'known '+prefix)
        out.append({k:int(v,10 if k in ['ps','case','cycle','causes','tick','state'] else 16) for k,v in raw.items()})
    return out

def review_transport(text,case_id,ctl):
    src=decoded(text,'OWNER_SOURCE');mgmt=decoded(text,'OWNER_MGMT');host=decoded(text,'OWNER_HOST')
    own=decoded(text,'OWN_TRACE');c=decoded(text,'CTL_TRACE')
    require(len(src)==len(own)==len(c) and len(src)>100,'paired source streams')
    require(all(b['ps']-a['ps']==10000 for a,b in zip(mgmt,mgmt[1:])),'continuous management sampling')
    gaps=[b['ps']-a['ps'] for a,b in zip(src,src[1:])]
    require(all(x==5000 or (case_id==4 and x>5000 and x%2500==0) for x in gaps),'source clock gaps')
    require(sum(x!=5000 for x in gaps)==int(case_id==4),'exact source-stop count')
    hgaps=[b['ps']-a['ps'] for a,b in zip(host,host[1:])]
    require(all(x==7300 or (case_id==3 and x>7300 and x%3650==0) for x in hgaps),'host clock gaps')
    require(sum(x!=7300 for x in hgaps)==int(case_id==3),'exact host-stop count')
    for s,o in zip(src,own):
        require(s['cold']==o['rst'] and s['ack']==(o['lost'] and o['invalid']) and s['lost']==o['lost'],'ACK from clocked actual owner')
        if s['cold']:
            require(not s['fault'],'no boundary fault')
            for k in range(8):
                a=(s['accepted']>>(64*k))&MAX;d=(s['delivered']>>(64*k))&MAX;z=(s['abandoned']>>(64*k))&MAX;p=(s['pending']>>(2*k))&3
                require(a==d+z+p,'retained boundary conservation')
                if k not in (2,5,7):require(z==0,'no request erasure')
    # Reconstruct ordered host requests/responses independently on both sides.
    specs=[('AW','awv','awr',['awa','awi']),('W','wv','wr',['wd','ws','wl']),('B','bv','br',['bi','bs']),
           ('CAW','cav','car',['ca']),('CW','cwv','cwr',['cwd','cws']),('CB','cbv','cbr',['cbs']),
           ('CAR','crv','crr',['cra']),('CR','rv','rr',['rd','rs'])]
    host_events={n:[] for n,*_ in specs};holds={}
    for h in host:
        if not h['rst']:holds={};continue
        for n,v,r,keys in specs:
            value=tuple(h[k] for k in keys)
            if n in holds:require(h[v] and holds.pop(n)==value,'host held '+n)
            if h[v] and not h[r]:holds[n]=value
            if h[v] and h[r]:host_events[n].append(value)
    core_events={}
    for name,v,r,keys in [('AW','awv','awr',['awa','awi']),('W','wv','wr',['wd','ws','wl']),('B','bv','br',['bi','bs'])]:
        core_events[name]=[tuple(o[k] for k in keys) for o in own if o['rst'] and o[v] and o[r]]
    for name,v,r,keys in [('CAW','av','ar',['aw']),('CW','wv','wr',['wd','ws']),('CB','bv','br',['bs']),('CAR','rv','rr',['ra']),('CR','ov','ore',['od','os'])]:
        core_events[name]=[tuple(x[k] for k in keys) for x,o in zip(c,own) if o['rst'] and x[v] and x[r]]
    require(core_events==host_events,'all source/host transactions agree including partial requests')
    events=[fields(x) for x in text.splitlines() if x.startswith('OWNER_EVENT ')]
    if case_id==0:
        require(not events and all(not x['loss'] for x in mgmt),'healthy baseline has no loss')
        return dict(reset_spans=[],host_events={k:len(v) for k,v in host_events.items()})
    cause={1:1,2:2,3:4,4:8,5:1,6:2,7:16,8:2}[case_id]
    require(events and events[0]['name']=={1:'link_pulse',2:'endpoint_pulse',3:'host_stop',4:'source_stop',5:'link_pulse',6:'endpoint_pulse',7:'pool_loss',8:'endpoint_pulse'}[case_id],'explicit loss stimulus')
    fault=int(events[0]['ps']);det=next(x for x in mgmt if x['loss']);invalid=next(x for x in src if x['invalid']);ack=next(x for x in src if x['ack'])
    require(fault<det['ps']<=fault+(800000 if case_id in (3,4) else 80000),'bounded management detection')
    require(ack['ps']>invalid['ps'],'clocked owner ACK after invalidation')
    if case_id==7:require(invalid['ps']<det['ps'],'pool loss starts in source')
    else:require(det['ps']<invalid['ps'],'host loss crosses to source')
    post=[x for x in mgmt if x['ps']>=det['ps']]
    require(all(x['loss'] and not x['ha'] and not x['sa'] and x['causes']==cause and x['tick']==det['tick'] for x in post),'sticky cause/tick and closed admission')
    resets=[x for x in post if x['req']]
    require(resets and resets[0]['ps']>=ack['ps']+20000,'reset waits returned real ACK')
    require(all(not x['hrst'] and not x['srst'] for x in resets),'shared reset assertion')
    released=next(x for x in post if x['ps']>resets[-1]['ps'] and x['hrst'] and x['srst'])
    lo=resets[0]['ps'];hi=resets[-1]['ps']
    nh=sum(lo<=x['ps']<=hi for x in host);ns=sum(lo<=x['ps']<=hi for x in src)
    require(nh>=64 and ns>=64,'both clocks see shared reset')
    require(post[-1]['state']==8 and post[-1]['ack'] and post[-1]['hrst'] and post[-1]['srst'],'restored interfaces without rearm')
    if case_id==4:
        resume=int(next(x['ps'] for x in events if x['name']=='source_resume'))
        require(all(not x['ack'] and not x['req'] and x['hrst'] and x['srst'] for x in post if x['ps']<resume),'source stop cannot authorize transport reset')
    return dict(detection_ps=det['ps']-fault,source_invalid_ps=invalid['ps'],owner_ack_ps=ack['ps'],reset_request_ps=lo,
                reset_spans=[dict(host_edges=nh,source_edges=ns,duration_ps=hi-lo+10000)],host_events={k:len(v) for k,v in host_events.items()})

def review(text):
    require(not re.search(r'^\s*(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    require(text.splitlines().count('OWNER_SUITE_PASS cases=9')==1,'one complete suite')
    groups={i:[] for i in CASES};current=None
    for line in text.splitlines():
        if line.startswith(('OWN_TRACE ','OWNER_SOURCE ','OWNER_HOST ','OWNER_MGMT ','OWNER_EVENT ','OWNER_PASS ')):
            current=int(fields(line)['case'])
        if current in groups:groups[current].append(line)
    results=[]
    for case,lines in groups.items():
        sub='\n'.join(lines);o=review_owner(sub,case);ctl=review_control(sub,1,case)
        results.append(dict(**o,control=ctl,transport=review_transport(sub,case,ctl)))
    return dict(status='PASS_REAL_NTRD_OWNER_GUARDED_CDC_9_CASES',cases=results)

if __name__=='__main__':
    import json,pathlib,sys
    print(json.dumps(review(pathlib.Path(sys.argv[1]).read_text()),indent=2))
