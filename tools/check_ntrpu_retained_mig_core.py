# Based on the frozen guarded MIG oracle. Source clock checks are explicitly
# parameterized; actual memory UI period and all byte/ownership checks remain.
# Guarded composition review: the retained boundary can accept core B/R
# immediately. Keep all cycle-by-cycle control/ownership comparisons unchanged;
# host-side B/R stalls are required by the separate transport checker.
# Only the direct-core B/R stall coverage requirement differs from the parent.
"""NTRD host frames through two actual MIGs: independent MMIO/ownership/DDR ledger."""
import re
from collections import Counter
from check_ntrpu_pcie_fragment_pool import FIELDS,fields,require,MAX

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
    require(not bv and not rv and aw is None and wd is None and write_receipt is None and read_address is None,'control transactions drained')
    require(not queued and running==bool(case_id) and reset_intervals==1,'retained running state after pool reset')
    for label,actual,keys in [('WRITE',writes,['addr','data','strb','resp']),('READ',reads,['addr','data','resp'])]:
        records=[fields(x) for x in text.splitlines() if x.startswith('CTL_'+label+' ')]
        require([int(x['n']) for x in records]==list(range(1,len(actual)+1)),'driver receipt count '+label)
        require(all(set(x)=={'n','cycle',*keys} for x in records),'driver receipt schema')
        require([tuple(int(x[k],16) for k in keys) for x in records]==actual,'driver receipts match actual AXI-Lite '+label)
    ends=[fields(x) for x in text.splitlines() if x.startswith('CTL_PASS ')]
    require(ends==[dict(writes=str(len(writes)),reads=str(len(reads)),seed=str(seed),cycles=str(len(rows)))],'control summary witnessed')
    require(reads[:3]==[(0,0x4452544e,0),(4,0x10000,0),(8,0xff,0)],'NTRD discovery profile')
    require(not snapshots,'fixture does not request snapshots')
    require(stats['accepted']==(18 if case_id==0 else 1) and stats['retired']==(0 if case_id else 18),'MMIO ownership inventory')
    require(stats['arm_stall']>0,'queued ARM pause exercised; core B/R may enter retained boundary immediately')
    if case_id:require(blocked['session']>0,'MMIO ACK rejects invalid session')
    return dict(writes=len(writes),reads=len(reads),rejected_writes=sum(w[-1]!=0 for w in writes),
                stats=dict(stats),blocked=dict(blocked),snapshots=snapshots,
                discovery=dict(id=0x4452544e,version=0x10000,capabilities=0xff,channels=2,address_bits=37))


def pattern(job,seed=7):return int.from_bytes(bytes((job*37+i*19+seed*7)&255 for i in range(64)),'little')
def address(j):
    if j in (13,17):return 0x100001040
    if j==14:return 0x100001080
    if j>=12:return 0x1fffffffc0
    return [0,0x100000040,0x800000000,0x1000000000,0x1800000000,0x1fffffffc0][j//2]

def review(text, source_period_ps):
    require(source_period_ps in (5000,37000), "approved source period")
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text),'simulator/model failure')
    prefix='NTRD case=0 '
    subset='\n'.join(x[len(prefix):] for x in text.splitlines() if x.startswith(prefix))
    ctl=review_control(subset,7,0)
    raw=[fields(x) for x in subset.splitlines() if x.startswith('OWN_TRACE ')]
    life=[fields(x) for x in subset.splitlines() if x.startswith('LIFE_TRACE ')]
    require(len(raw)==len(life) and [int(x['cycle']) for x in raw]==[int(x['cycle']) for x in life]==list(range(1,len(raw)+1)),'complete source trace from first clock')
    require(all(set(x)==set(FIELDS) for x in raw),'ownership trace schema')
    life_schema='cycle ps cold poolreset memreset admit restart synced rearm_ready calibrated ui_rst source_run memory_run banks pending external rearm'.split()
    require(all(set(x)==set(life_schema) for x in life),'complete lifetime schema')
    require([int(x['ps']) for x in life]==list(range(source_period_ps//2,source_period_ps//2+source_period_ps*len(life),source_period_ps)),'source timestamps from time zero')
    first_both=next(i for i,x in enumerate(life) if x['calibrated']=='3' and x['ui_rst']=='0')
    require(all(x['cold']==x['poolreset']==x['admit']=='0' for x in life[:first_both]),'cold barrier before both actual MIGs')
    one_ready=sum(x['calibrated']=='1' and int(x['ui_rst'],16)&1==0 for x in life)
    require(one_ready*source_period_ps>=2500000-source_period_ps,'staggered MIG start duration')
    begin=[fields(x) for x in text.splitlines() if x.startswith('MANAGED_MIG_BEGIN ')]
    ends=[fields(x) for x in text.splitlines() if x.startswith('MANAGED_MIG_PASS ')]
    require(begin==[dict(jobs='18',first='00000000ffffffff')] and len(ends)==1,'one fixture header and finish')
    end=ends[0];require(set(end)==set('arms lines commands data cpl done acks aw w b reads frames rejects one_ready end_ps'.split()),'terminal schema')
    jobs=[int(fields(x)['job']) for x in subset.splitlines() if x.startswith('JOB ')]
    require(jobs==list(range(18)),'ordered host jobs')
    schema=dict(CLOCK='ps channel period phase',AW='ps job channel address len id size burst cache',W='ps job channel data strb last',B='ps job channel id status',AR='ps channel index identity address',R='ps channel index identity data status id last')
    events=[]
    for x in text.splitlines():
        if not x.startswith('MANAGED_MIG_'):continue
        kind=x.split()[0][len('MANAGED_MIG_'):]
        if kind in ('BEGIN','PASS'):continue
        f=fields(x);require(kind in schema and set(f)==set(schema[kind].split()),'memory event schema')
        require(all(re.fullmatch('[0-9a-f]+',v) for v in f.values()),'known memory fields')
        e={k:int(v,10 if k in ('ps','job','channel','index','identity','period','phase') else 16) for k,v in f.items()}
        events.append((kind,e))
    require([e['ps'] for _,e in events]==sorted(e['ps'] for _,e in events),'memory chronology')
    clocks={e['channel']:e for kind,e in events if kind=='CLOCK'}
    require(len([1 for k,e in events if k=='CLOCK'])==2 and set(clocks)=={0,1},'two actual UI clocks')
    require(all(e['period']==5000 and e['ps']%5000==e['phase'] for e in clocks.values()) and (clocks[1]['phase']-clocks[0]['phase'])%5000==731,'UI period and independent phase')
    mem=[(k,e) for k,e in events if k in ('AW','W','B')];by_mem={j:{} for j in jobs}
    for kind,e in events:
        ch=e['channel'];require(ch in clocks and e['ps']%5000==clocks[ch]['phase'],'actual memory clock edge')
    for kind,e in mem:
        j=e['job'];require(j in by_mem and kind not in by_mem[j] and e['channel']==j%2,'owned memory event')
        if kind=='AW':require((e['address'],e['len'],e['id'],e['size'],e['burst'],e['cache'])==(address(j),0,0,6,1,0),'full37 address and AXI attributes')
        if kind=='W':require((e['data'],e['strb'],e['last'])==(pattern(j),MAX,1),'all memory write bytes')
        if kind=='B':require(e['id']==e['status']==0,'actual MIG write response')
        by_mem[j][kind]=e
    seed=7;case_id=0;fault_ps=None
    counts=Counter({k:0 for k in ('arms','lines','commands','data','cpl','done','acks','frames','rejects','publications')})
    ctx=None;contexts={};epoch=0;cookie=0xffffffff;seen=False;lost=False;active=False;started=False;hold={}
    aw=wd=None;pending=None;prior_b=False;rejects=0;frame_index=Counter();reset_rows=0;metrics=[]
    fixture={j:([(0,0,0)] if j==12 else [(0,0,0),(0,1,2)] if j==13 else [(0,0,0),(1,0,0)]) for j in jobs}
    final=None
    for row,lr in zip(raw,life):
        require(lr['cold']==row['rst'] and lr['admit']==row['admit'] and lr['synced']==row['restart'],'paired reset/admission evidence')
        if lr['restart']=='1':require(all(lr[k]=='0' for k in ('poolreset','memreset','admit','rearm_ready')),'restart closes every operational reset/admission')
        if row['rst']=='0':
            require(not started, 'cold lifetime never follows runtime pool reset')
            require(all(row[k]=='0' for k in ['ar','kr','rc','awr','wr','cv','dv','ov']), 'cold quiescence')
            reset_rows+=1;continue
        started=True
        require(lr['restart']==lr['synced']=='0' and lr['admit']=='1' if seen else lr['restart']==lr['synced']=='0','one uninterrupted cold session')
        require(all(re.fullmatch('[0-9a-f]+',v) for k,v in row.items() if k!='case'),'known active ownership')
        s={k:int(v,10 if k in ('cycle','case') else 16) for k,v in row.items()}
        l={k:int(v,10 if k in ('cycle','ps') else 16) for k,v in lr.items()};now=l['ps'];job=s['case']
        require(s['epoch']==s['be']==epoch and s['held']==bool(ctx) and s['bh']==active,'retained collector/control/bridge identity')
        require(s['next']==cookie and not s['exhausted'],'cookie lifetime')
        invalid=bool(lost or s['restart'] or (seen and not s['admit']))
        require(s['lost']==lost and s['invalid']==invalid,'session loss is sticky')
        require(s['fault']==s['pf']==s['rejected']==0,'no unrelated pool/protocol fault')
        require(s['ar']==bool(s['car'] and s['bar']) and s['rc']==bool(s['kv'] and s['kr']),'joint ARM and ACK')
        barrier=s['stopped'] and s['isolated'] and s['qe']==epoch
        require(s['kr']==bool(s['rv'] and s['ke']==s['re'] and s['kt']==s['rt'] and barrier and s['ti'] and not s['awv'] and not s['wv'] and s['rr'] and not invalid),'full ACK predicate')
        if l['poolreset']==0 and fault_ps is not None and now>fault_ps:
            require(invalid and active and not l['admit'],'runtime reset invalidates, retains lease')
        for name,v,r,ks in [('B','bv','br',['bi','bs']),('LINE','lv','lr',['le','lt','la','ld']),
                             ('CMD','cv','cr',['cc','ca','ct','cl']),('DATA','dv','dr',['data']),
                             ('DONE','ov','ore',['oe','ot','os','ops']),('RESULT','rv','rc',['re','rt','rs'])]:
            value=tuple(s[k] for k in ks)
            if name in hold:require(s[v] and hold.pop(name)==value,'stable held '+name)
            if s[v] and not s[r]:hold[name]=value
        if s['av'] and s['ar']:
            require(not active and not invalid and s['admit'] and s['drained'] and barrier and s['ti'] and not s['awv'] and not s['wv'],'ARM actual supervisor gate')
            require(job in jobs and job not in contexts,'one ARM per job')
            require((s['ae'],s['at'],s['aw'],s['aa'],s['ac'])==(0x100000001+job,0x800000010000abcd,0x1000fffc0,address(job),job%2),'ARM full descriptor')
            ctx=dict(epoch=s['ae'],tag=s['at'],address=s['aa'],channel=s['ac'],arm=now,line=None,cmd=None,data=None,cpl=None,done=None,result=None,ack=None,cookie=None)
            contexts[job]=ctx;active=True;epoch=s['ae'];counts['arms']+=1
        if s['lv']:
            require(ctx and active and ctx['line'] is None and not invalid,'one assembled LINE in live session')
            require((s['le'],s['lt'],s['la'],s['ld'],s['bytes'])==(epoch,ctx['tag'],ctx['address'],pattern(job,seed),MAX),'complete collected payload')
            if s['lr']:ctx['line']=now;ctx['cookie']=cookie;cookie+=1;counts['lines']+=1
        if s['cv']:
            require(ctx and ctx['line'] is not None and ctx['cmd'] is None,'CMD owned by LINE')
            require((s['cc'],s['ca'],s['ct'],s['cl'])==(ctx['channel'],ctx['address'],ctx['cookie'],0),'pool command/cookie')
            if s['cr']:ctx['cmd']=now;counts['commands']+=1
        if s['dv']:
            require(ctx and ctx['cmd'] is not None and ctx['cmd']<now and ctx['data'] is None and s['data']==pattern(job,seed),'ordered pool data')
            if s['dr']:ctx['data']=now;counts['data']+=1
        for ch in range(2):
            if (s['pv']&s['pr'])>>ch&1:
                require(ctx and ctx['cpl'] is None and ch==ctx['channel'] and (s['pt']>>(ch*64))&MAX==ctx['cookie'] and (s['ps']>>(ch*2))&3==0,'owned completion cookie/channel')
                require('B' in by_mem[job] and by_mem[job]['B']['ps']<now,'completion after actual memory response')
                ctx['cpl']=now;counts['cpl']+=1
        if s['ov']:
            require(ctx and ctx['cpl'] is not None and ctx['cpl']<now and ctx['done'] is None and not invalid and s['drained'],'DONE follows drained memory')
            require((s['oe'],s['ot'],s['os'],s['ops'],s['err'])==(epoch,ctx['tag'],int(job in (15,16)),0,0),'DONE original identity')
            if s['ore']:ctx['done']=now;counts['done']+=1
        if s['rv']:
            status=1 if job in (12,15,16) else 4 if job==13 else 0
            require(ctx and (s['re'],s['rt'],s['rs'])==(epoch,ctx['tag'],status),'retained RESULT context')
            if ctx['result'] is None:
                require(barrier and s['ti'] and not s['awv'] and not s['wv'],'RESULT external proof/drain')
                require(ctx['done'] is not None or ctx['line'] is None,'no false completion')
                ctx['result']=now;counts['publications']+=1
        if s['rc']:
            require(ctx and active and ctx['result'] is not None and ctx['ack'] is None and not invalid and s['drained'] and aw is None and wd is None and pending is None,'atomic ACK only after all drains')
            ctx['ack']=now;active=False;counts['acks']+=1
        if aw is not None:require(not s['awr'],'accepted frame AW retained until W')
        if wd is not None:require(not s['wr'],'accepted frame W retained until AW')
        if s['bv']:require(not s['awr'] and not s['wr'],'response backpressure closes frame ingress')
        if s['awv'] and s['awr']:
            require(aw is None and pending is None,'one ingress AW')
            require((s['awa'],s['awi'],s['awl'],s['aws'],s['awb'],s['awk'])==(0x1000fffc0,job%16,0,6,1,0),'frame AW control')
            aw=job
        if s['wv'] and s['wr']:
            require(wd is None and pending is None and s['ws']==MAX and s['wl']==1,'one full frame W')
            wd=s['wd']
        if aw is not None and wd is not None:
            require(aw==job and ctx and frame_index[job]<len(fixture[job]),'frame under retained context')
            half,conflict,resp=fixture[job][frame_index[job]];payload=pattern(job,seed)^conflict
            expected=0x313046555052544e | epoch<<64 | ctx['tag']<<128 | (half*32)<<192 | ((payload>>(half*256))&((1<<256)-1))<<256
            require(wd==expected,'F01 header and all bytes across reset')
            pending=(job,resp);frame_index[job]+=1;aw=wd=None
        if s['bv']:
            require(pending and (s['bi'],s['bs'])==(pending[0]%16,pending[1]),'B for accepted ingress frame')
            if not prior_b and pending[1]:rejects+=1
            if s['br']:
                counts['frames']+=1;counts['rejects']+=int(pending[1]!=0);pending=None
        require(s['frames']==rejects and s['identity']==0,'global rejection ledger retained')
        prior_b=bool(s['bv'] and not s['br']);seen=seen or bool(s['admit']);lost=invalid;final=(s,l)
    require(reset_rows>40 and started,'startup reset interval exercised')

    expected=dict(arms=18,lines=16,commands=16,data=16,cpl=16,done=16,acks=18,frames=35,rejects=1,publications=18)
    require(dict(counts)==expected,'all source ownership obligations')
    require(all(int(end[k])==v for k,v in expected.items() if k!='publications'),'terminal source inventory')
    require(frame_index==Counter({j:len(v) for j,v in fixture.items()}),'all expected F01 host frames')
    require(aw is None and wd is None and pending is None and not hold,'all host obligations drained')
    for j,ctx in contexts.items():
        x=by_mem[j];require(set(x)==(set() if j in (12,13) else {'AW','W','B'}),'memory write conservation per job')
        require(ctx['result'] is not None and ctx['ack'] is not None,'every result retired by MMIO')
        if x:
            require(ctx['cmd']<x['AW']['ps']<x['B']['ps'] and ctx['data']<x['W']['ps']<x['B']['ps']<ctx['cpl']<ctx['done']<=ctx['result']<ctx['ack'],'memory/completion/retirement causality')
            metrics.append(dict(job=j,channel=j%2,line_to_b_ps=x['B']['ps']-ctx['line'],b_to_cpl_ps=ctx['cpl']-x['B']['ps'],line_to_ack_ps=ctx['ack']-ctx['line']))
    golden=[{},{}];waiting=[None,None];read_order=[[],[]];readbacks=[]
    for kind,e in events:
        ch=e['channel']
        if kind=='B':
            j=e['job'];golden[ch][address(j)]=dict(identity=j,data=pattern(j),b=e['ps'])
        if kind=='AR':
            require(waiting[ch] is None and e['address'] in golden[ch] and e['address']==address(e['identity']),'AR to independently written location')
            g=golden[ch][e['address']]
            require(g['identity']==e['identity'] and g['b']<e['ps'] and e['index']==len(read_order[ch])+1,'readback overwrite version')
            waiting[ch]=dict(identity=e['identity'],index=e['index'],data=g['data'],ar=e['ps'])
        if kind=='R':
            p=waiting[ch];require(p is not None,'one R for accepted AR')
            require((e['identity'],e['index'],e['data'],e['status'],e['id'],e['last'])==(p['identity'],p['index'],p['data'],0,0,1),'readback all64 bytes and attributes')
            require(p['ar']<e['ps'],'read response ordering')
            read_order[ch].append(p['identity']);readbacks.append(dict(channel=ch,identity=p['identity'],ar_to_r_ps=e['ps']-p['ar']));waiting[ch]=None
    require(read_order==[[0,2,4,6,8,10,10,0,2,4,6,8,16,14],[1,3,5,7,9,11,11,1,3,5,7,9,15,17]],'readback coverage for STOP/conflict/no-write/overwrites')
    require(waiting==[None,None] and [len(g) for g in golden]==[7,7],'read obligations drained')
    require(int(end['aw'])==int(end['w'])==int(end['b'])==16 and int(end['reads'])==28 and int(end['one_ready'])==one_ready,'memory/ready summary')
    s,l=final
    require(not active and ctx['ack'] and s['idle'] and s['drained'] and not l['banks'] and not l['pending'] and not s['invalid'] and s['next']==0x10000000f,'terminal live drained session')
    require(reset_rows*source_period_ps>5000000 and started and len(raw)==int(l['cycle']) and int(end['end_ps'])>l['ps'],'cold trace and finish')
    return dict(status='PASS_NTRD_HOST_TWO_MIG_BYTES',counts=dict(counts),control=ctl,source_samples=len(raw),write_bytes=1024,read_bytes=1792,
                memory_events=len(mem),latencies=metrics,readback_latencies=readbacks,clocks=clocks,
                scope='NTRD AXI-Lite and F01 host BFMs through retained collector/bridge, actual supervisor/pool/two CDCs and two generated MIG/generic RDIMM models. One cold session/exclusive lease. Not native PCIe, exact commercial DIMMs, warm reset, routed timing, hardware throughput or inference.')

from decimal import Decimal
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING,WARNING_DELAY,BEAT_PS,records
need=require
from check_ntrpu_fragment_dual_mig import rank_witness

def periodic(text):
    parts = [[],[]]; candidates = {}
    for ch,record in re.findall(r'^DUAL_PER ([01]) (.*)$',text,re.M): parts[int(ch)].append(record)
    need(len(re.findall(r'^DUAL_PER ',text,re.M))==sum(map(len,parts)),'Known periodic port tags')
    for ch,lines in enumerate(parts):
        for pin in records('\n'.join(lines),'NTRPU_DDR_PIN_CMD', ['ps','slot','rank','act_n','cke'], ['cmd','adr','bg','bank']):
            if pin['act_n']==1 and (pin['adr']>>14)&7==5:
                for beat in range(8):candidates.setdefault(pin['ps']+WARNING_DELAY+beat*BEAT_PS,set()).add(ch)
    warning_counts = [0,0]
    for line in text.splitlines():
        if re.search(r':\s*WARNING\s*:',line):
            match=WARNING.fullmatch(line);need(match is not None,'Unknown model warning')
            owners=candidates.get(int(match[7]),set());need(len(owners)==1,'Unique physical owner of model warning')
            ch=next(iter(owners));parts[ch].append(line);warning_counts[ch]+=1
    finish=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    times={int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in finish}
    need(len(times)==1,'One global finish time');finish_ps=times.pop()
    result=[]
    for ch,lines in enumerate(parts):
        local='\n'.join(lines); witness=review_periodic(local)
        rows=records(local,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
        need(len(rows)==1,'One final periodic drain per controller');d=rows[0]
        phase=(witness['configuration']['ps']+2500)%5000
        first=d['begin_ps']+(phase-d['begin_ps'])%5000
        if first==d['begin_ps']:first+=5000
        need(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'Consecutive local drain edges')
        need(0<=finish_ps-d['end_ps']<1000000,'Global finish within bounded peer drain window')
        counts=witness['counters']
        need(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'Drain counters')
        returns=records(local,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
        need(returns[-1]['ps']<d['end_ps'] and all(x['return_ps']<d['end_ps'] for x in witness['lifecycle']),'All returns before local drain')
        need(counts['user_cas']==(15 if ch==0 else 14) and witness['physical_writes']==(9 if ch==0 else 8),'Per-controller physical transfers including job18')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='DUAL_PERIODIC_WITNESS_PASS',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)

