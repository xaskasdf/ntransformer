"""Independent ownership and MMIO oracle for the NTRD managed composition.

Reconstructs full frames, payloads, lease lifetime and transport cookies. Does
not use DUT phase encodings to decide ownership or successful completion.
"""
import re
from collections import Counter

FIELDS='cycle case rst stop stopped isolated qe admit restart pf drained av ar car bar ae at aw aa ac kv kr ke kt rr awv awr awa awi awl aws awb awk wv wr wd ws wl bv br bi bs lv lr le lt la ld cv cr cc ca ct cl dv dr data pv pr pt ps ov ore oe ot os ops err rv rc re rt rs epoch held be bh idle phase fail error bytes ti fault lost invalid exhausted next rejected frames identity code faults'.split()
MAX=(1<<64)-1
CASES=list(range(1,13))
LINE_CASES={1,4,5,6,7,8,9,11}
DONE={1:0,4:0,5:1,6:1,7:4,8:1,9:0}
RESULT={1:0,2:1,3:4,4:0,5:1,6:1,7:5,8:6,9:0,10:1}
FRAMES={1:[(0,0,0),(1,0,0),(0,0,2)],2:[(0,0,0)],3:[(0,0,0),(0,1,2)],
        4:[(0,2,2),(0,0,0),(1,0,0)],5:[(0,0,0),(1,0,0)],6:[(0,0,0),(1,0,0)],
        7:[(0,0,0),(1,0,0)],8:[(0,0,0),(1,0,0),(0,0,2)],9:[(0,0,0),(1,0,0)],
        10:[(0,0,0)],11:[(0,0,0),(1,0,0)]}

def require(ok,message):
    if not ok:raise ValueError(message)

def fields(line):
    pairs=re.findall(r'(\w+)=([^\s]+)',line)
    require(len(pairs)==len(dict(pairs)), 'duplicate field')
    return dict(pairs)

def payload(job,seed):
    return int.from_bytes(bytes((job*37+k*19+seed*7)&255 for k in range(64)), 'little')

def review_ownership(text,seed):
    require(not re.search(r'^\s*(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    rows=[fields(x) for x in text.splitlines() if x.startswith('OWN_TRACE ')]
    begin=[fields(x) for x in text.splitlines() if x.startswith('OWN_BEGIN ')]
    cases=[fields(x) for x in text.splitlines() if x.startswith('OWN_CASE ')]
    ends=[fields(x) for x in text.splitlines() if x.startswith('OWN_PASS ')]
    require(begin==[dict(seed=str(seed),first='00000000ffffffff')], 'one exact BEGIN')
    require(cases==[dict(id=str(j),seed=str(seed)) for j in CASES], 'ordered fixture cases')
    require(len(ends)==1,'one PASS')
    require(rows and [int(x['cycle']) for x in rows]==list(range(1,len(rows)+1)), 'complete trace from cycle1')
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
            expected=3 if job==7 else 0
            require(s['ps']==expected<<(2*ctx['channel']), 'fixture completion status')
            ctx['cpl']=cycle;event('completions')
        if s['ov']:
            require(active and ctx and ctx['cpl'] is not None and ctx['cpl']<cycle and ctx['done'] is None, 'DONE requires actual completion')
            require((s['oe'],s['ot'],s['os'],s['ops'])==(ctx['epoch'],ctx['tag'],DONE[job],3 if job==7 else 0),'DONE identity/status')
            require(not invalid and s['drained'], 'drained live DONE in this fixture')
            if s['ore']:ctx['done']=cycle;event('done')
        if s['rv']:
            require(ctx and (active or ctx['acked']), 'retained result context')
            require((s['re'],s['rt'],s['rs'])==(ctx['epoch'],ctx['tag'],RESULT[job]),'visible RESULT full identity/status')
            if ctx['result'] is None:
                require(barrier and transport, 'RESULT external epoch barrier and drained AXI')
                require(ctx['done'] is not None or job in (2,3,10), 'no result before destination completion')
                ctx['result']=cycle;event('publications')
        if s['kr']:
            require(active and ctx and ctx['result'] is not None and not invalid and s['drained'] and not s['pv'], 'ACK cannot discard active/invalid destination')
            require(ctx['done'] is not None or job in (2,3,10), 'ACK completion or canceled partial line')
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
    require(observed_cases==CASES and resets==4, 'complete case/reset sequence')
    require(frame_index==Counter({j:len(v) for j,v in FRAMES.items()}),'all fixture frames present')
    expected=dict(arms=11,lines=8,commands=8,data=8,done=7,acks=8,frames=23,rejects=4,completions=7,publications=10)
    require(dict(counts)==expected,'complete transaction inventory')
    for j,c in bycase.items():
        require(c['arms']==int(j<=11) and c['lines']==int(j in LINE_CASES) and c['acks']==int(j<=8),'per-case ownership inventory')
        require(c['commands']==c['data']==int(j in LINE_CASES) and c['done']==int(j in DONE),'per-case transfer inventory')
    require(ends[0]==dict(seed=str(seed),cycles=str(len(rows)),**{k:str(expected[k]) for k in ['arms','lines','commands','data','done','acks','frames','rejects']}),'PASS counters match observed events')
    require(all(stalls[x]>0 for x in ['B','command','data','result']),'intentional backpressure exercised')
    require(final['invalid']==1 and final['bh']==final['held']==final['av']==0 and not final['rv'] and not final['ov'],'invalid idle session refuses new ARM')
    require(aw is None and wd is None and pending is None,'final ingress drain')
    return dict(seed=seed,cycles=len(rows),counts=dict(counts),per_case={j:dict(c) for j,c in bycase.items()},
                blocked_ack_cycles=dict(blocked),stalls=dict(stalls),metrics=metrics,reset_intervals=resets,
                scope='NTRD MMIO/collector/bridge; exclusive pool protocol peers, no MIG/PCIe/CDC/physical timing.')

CONTROL_FIELDS='cycle aw av ar wd ws wv wr bv br bs ra rv rr ov ore od os irq fault'.split()

def review_control(text,seed):
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
    require(not bv and not rv and aw is None and wd is None and write_receipt is None and read_address is None,'control transactions drained')
    require(not queued and not running and reset_intervals==4,'final invalid idle profile state')
    for label,actual,keys in [('WRITE',writes,['addr','data','strb','resp']),('READ',reads,['addr','data','resp'])]:
        records=[fields(x) for x in text.splitlines() if x.startswith('CTL_'+label+' ')]
        require([int(x['n']) for x in records]==list(range(1,len(actual)+1)),'driver receipt count '+label)
        require(all(set(x)=={'n','cycle',*keys} for x in records),'driver receipt schema')
        require([tuple(int(x[k],16) for k in keys) for x in records]==actual,'driver receipts match actual AXI-Lite '+label)
    ends=[fields(x) for x in text.splitlines() if x.startswith('CTL_PASS ')]
    require(ends==[dict(writes=str(len(writes)),reads=str(len(reads)),seed=str(seed),cycles=str(len(rows)))],'control summary witnessed')
    require(reads[:12]==[(0,0x4452544e,0),(4,0x10000,0),(8,0xff,0),(0x9c,0x5052544e,0),(0xa0,0x31304655,0),
                           (0xa4,64,0),(0xa8,32,0),(0xac,0,0),(0xb0,0,0),(0xb4,0,0),(0xbc,2,0),(0xc0,37,0)], 'complete discovery profile')
    require(snapshots==[(1,0,6),(3,1,0x107),(4,1,6)],'frozen telemetry snapshot inventory')
    require(all(blocked[k]>0 for k in ('identity','barrier','transport','drain','session','channel_range','channel_queued','channel_running','unavailable_arm')),'MMIO rejection coverage')
    require(stats['accepted']==11 and stats['retired']==8 and stats['b_stall']>100 and stats['r_stall']>40 and stats['arm_stall']>10,'MMIO coverage and backpressure')
    return dict(writes=len(writes),reads=len(reads),rejected_writes=sum(w[-1]!=0 for w in writes),
                stats=dict(stats),blocked=dict(blocked),snapshots=snapshots,
                discovery=dict(id=0x4452544e,version=0x10000,capabilities=0xff,channels=2,address_bits=37))

def review(text,seed):
    return dict(ownership=review_ownership(text,seed),control=review_control(text,seed))
