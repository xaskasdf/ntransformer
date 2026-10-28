"""Managed ownership through real supervisor/pool/CDC resets; memory peers only."""
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
    require(stats['accepted']==(4 if case_id==0 else 1) and stats['retired']==(0 if case_id else 4),'MMIO ownership inventory')
    require(stats['b_stall']>0 and stats['r_stall']>0 and stats['arm_stall']>0,'MMIO pauses exercised')
    if case_id:require(blocked['session']>0,'MMIO ACK rejects invalid session')
    return dict(writes=len(writes),reads=len(reads),rejected_writes=sum(w[-1]!=0 for w in writes),
                stats=dict(stats),blocked=dict(blocked),snapshots=snapshots,
                discovery=dict(id=0x4452544e,version=0x10000,capabilities=0xff,channels=2,address_bits=37))

def pattern(job,seed):return int.from_bytes(bytes((job*37+i*19+seed*7)&255 for i in range(64)),'little')

def review_case(text,case_id):
    seed=case_id+1
    ctl=review_control(text,seed,case_id)
    raw=[fields(x) for x in text.splitlines() if x.startswith('OWN_TRACE ')]
    life=[fields(x) for x in text.splitlines() if x.startswith('LIFE_TRACE ')]
    require(len(raw)==len(life) and [int(x['cycle']) for x in raw]==[int(x['cycle']) for x in life]==list(range(1,len(raw)+1)), 'complete paired cycle trace')
    require(all(set(x)==set(FIELDS) for x in raw),'ownership trace schema')
    life_schema='cycle ps cold poolreset memreset admit restart synced rearm_ready calibrated ui_rst source_run memory_run banks pending'.split()
    require(all(set(x)==set(life_schema) for x in life),'lifetime trace schema')
    faults=[fields(x) for x in text.splitlines() if x.startswith('FAULT ')]
    ends=[fields(x) for x in text.splitlines() if x.startswith('CASE_PASS ')]
    jobs=[int(fields(x)['job']) for x in text.splitlines() if x.startswith('JOB ')]
    require(len(ends)==1 and len(faults)==int(case_id!=0), 'one terminal and expected fault')
    require(jobs==([0,1,2,3] if case_id==0 else [case_id%2]),'case job inventory')
    end=ends[0];fault_ps=int(faults[0]['ps']) if faults else None
    require(set(end)==set('ps job held invalid epoch next arms lines commands data cpl done acks frames rejects'.split()),'terminal schema')
    require(int(end['job'])==jobs[-1],'terminal job')
    times=[int(x['ps']) for x in life]
    require(times[0]==2500 and all(b>a for a,b in zip(times,times[1:])),'monotonic source timestamps')
    gaps=[b-a for a,b in zip(times,times[1:])]
    require(sum(x!=5000 for x in gaps)==int(case_id==3),'only intentional source-clock pause')
    if case_id==3:require(max(gaps)==1802500,'source clock stopped for injected interval')
    initial=[x for x in life if int(x['ps'])<2240000]
    require(len(initial)>400 and all(x['cold']==x['poolreset']==x['admit']=='0' for x in initial),'no startup before both channels ready')
    if case_id:
        f=faults[0]
        require(set(f)==set('ps job epoch cookie result status'.split()),'fault schema')
        before=max(i for i,t in enumerate(times) if t<fault_ps)
        for k,own in [('epoch','epoch'),('cookie','next'),('result','rv'),('status','rs')]:
            require(int(f[k],16)==int(raw[before][own],16),'fault marker matches observed '+k)
        require(int(f['job'])==jobs[0] and int(raw[before]['case'])==jobs[0],'fault job identity')
        window=[x for x in life if fault_ps<int(x['ps'])<fault_ps+1800000]
        if case_id in (1,7):require(window and all(x['calibrated']==('1' if case_id==1 else '2') for x in window),'calibration loss interval')
        if case_id==4:require(window and all(x['memory_run']=='1' for x in window),'other memory clock halted')
        if case_id==5:require(window and all(x['ui_rst']=='1' for x in window),'other UI reset interval')
        ct=[fields(x) for x in text.splitlines() if x.startswith('CTL_TRACE ')]
        after=next(i for i,t in enumerate(times) if t>fault_ps)
        if case_id==4:require(ct[after]['ar']=='0' and ct[after]['wr']=='1' and ct[after]['bv']=='0','pending MMIO AW without W across fault')
        if case_id==5:require(ct[after]['ov']=='1' and ct[after]['ore']=='0' and int(ct[after]['od'],16)==0x80000001,'held MMIO R across fault')
        if case_id==6:require(raw[after]['awr']=='0' and raw[after]['wr']=='1' and raw[after]['bv']=='0','pending frame AW without W across fault')
        if case_id==7:require(ct[after]['bv']=='1' and ct[after]['br']=='0' and ct[after]['bs']=='0','held MMIO B across fault')
    mem=[]
    for x in text.splitlines():
        if x.startswith(('MEM_AW ','MEM_W ','MEM_B ')):
            kind=x.split()[0][4:];f=fields(x)
            require(set(f)==set('ps job channel'.split())|set({'AW':'address len id size burst cache','W':'data strb last','B':'id status'}[kind].split()),'memory event schema')
            require(all(re.fullmatch('[0-9a-f]+',v) for v in f.values()), 'known memory event')
            e={k:int(v,10 if k in ('ps','job','channel') else 16) for k,v in f.items()}
            mem.append((kind,e))
    by_mem={j:{} for j in jobs}
    for kind,e in mem:
        j=e['job'];require(j in by_mem and kind not in by_mem[j], 'no duplicated memory transfer')
        require(e['channel']==j%2,'memory channel selected explicitly')
        if kind=='AW':require((e['address'],e['len'],e['id'],e['size'],e['burst'],e['cache'])==
                             (0x1fffffffc0 if j%2 else 0x100000040,0,0,6,1,0),'memory address geometry/control')
        if kind=='W':require((e['data'],e['strb'],e['last'])==(pattern(j,seed),MAX,1),'all actual memory payload bytes')
        if kind=='B':require(e['id']==e['status']==0,'memory response context')
        by_mem[j][kind]=e
    counts=Counter({k:0 for k in ('arms','lines','commands','data','cpl','done','acks','frames','rejects','publications')})
    ctx=None;contexts={};epoch=0;cookie=0xffffffff;seen=False;lost=False;active=False;started=False;hold={}
    aw=wd=None;pending=None;prior_b=False;rejects=0;frame_index=Counter();reset_rows=0;metrics=[]
    fixture={}
    for j in jobs:
        if case_id==0:
            fixture[j]=[(0,0,0),(1,0,0)] if j in (0,3) else [(0,0,0)] if j==1 else [(0,0,0),(0,1,2)]
        else:fixture[j]=[(0,0,0),(1,0,0)] if case_id in (2,3,4,5) else [(0,0,int(case_id==6)*2)]
    final=None
    for row,lr in zip(raw,life):
        require(lr['cold']==row['rst'] and lr['admit']==row['admit'] and lr['synced']==row['restart'],'paired reset/admission evidence')
        if lr['restart']=='1':require(all(lr[k]=='0' for k in ('poolreset','memreset','admit','rearm_ready')),'restart closes every operational reset/admission')
        if row['rst']=='0':
            require(not started, 'cold lifetime never follows runtime pool reset')
            require(all(row[k]=='0' for k in ['ar','kr','rc','awr','wr','cv','dv','ov']), 'cold quiescence')
            reset_rows+=1;continue
        started=True
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
            require((s['ae'],s['at'],s['aw'],s['aa'],s['ac'])==(0x100000001+job,0x800000010000abcd,0x1000fffc0,0x1fffffffc0 if job%2 else 0x100000040,job%2),'ARM full descriptor')
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
            require((s['oe'],s['ot'],s['os'],s['ops'],s['err'])==(epoch,ctx['tag'],0,0,0),'DONE original identity')
            if s['ore']:ctx['done']=now;counts['done']+=1
        if s['rv']:
            status=(1 if job==1 else 4 if job==2 else 0) if case_id==0 else 0 if case_id in (3,4,5) else 1
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
    expected=dict(arms=4 if case_id==0 else 1,lines=2 if case_id==0 else int(case_id in (2,3,4,5)),
                  commands=2 if case_id==0 else int(case_id in (2,3,4,5)),data=2 if case_id==0 else int(case_id in (2,3,4,5)),
                  cpl=2 if case_id==0 else int(case_id in (3,4,5)),done=2 if case_id==0 else int(case_id in (3,4,5)),
                  acks=4 if case_id==0 else 0,frames=7 if case_id==0 else 2 if case_id in (2,3,4,5) else 1,
                  rejects=int(case_id in (0,6)),publications=4 if case_id==0 else int(case_id!=2))
    require(dict(counts)==expected,'complete owned transfer inventory')
    require(all(int(end[k])==v for k,v in expected.items() if k!='publications'),'terminal observed counts')
    require(frame_index==Counter({j:len(v) for j,v in fixture.items()}),'all expected host frames')
    require(aw is None and wd is None and pending is None,'accepted ingress obligations all drained')
    for j,ctx in contexts.items():
        transfers=by_mem[j]
        require(set(transfers)==({'AW','W','B'} if ctx['cpl'] is not None else {'AW','W'} if ctx['data'] is not None else set()),'actual pool/MIG-boundary conservation')
        if transfers:
            require(ctx['cmd']<transfers['AW']['ps'] and ctx['data']<transfers['W']['ps'],'CDC transport follows source events')
            if 'B' in transfers:
                require(max(transfers[k]['ps'] for k in ('AW','W'))<transfers['B']['ps']<ctx['cpl'],'actual response order')
                metrics.append(dict(job=j,line_to_b_ps=transfers['B']['ps']-ctx['line'],b_to_cpl_ps=ctx['cpl']-transfers['B']['ps']))
    s,l=final
    require(int(end['held'])==s['bh']==bool(case_id) and int(end['invalid'])==s['invalid']==bool(case_id),'terminal lifetime')
    require(int(end['epoch'],16)==s['epoch'] and int(end['next'],16)==s['next'],'terminal ledger matches trace')
    require(l['cycle']==len(raw) and int(end['ps'])>l['ps'],'terminal follows final sample')
    if case_id:
        require(l['poolreset']==l['memreset']==l['admit']==l['rearm_ready']==0 and l['restart']==l['synced']==1,'no reopening after restored inputs')
        require(int(end['ps'])-fault_ps>3600000,'restore/rearm observation plus MMIO rejection')
        require(l['calibrated']==3 and l['ui_rst']==0 and l['memory_run']==3 and l['source_run']==1,'inputs restored but session remains invalid')
        raw_i=next(i for i,x in enumerate(life) if x['restart']=='1');sync_i=next(i for i,x in enumerate(life) if x['synced']=='1')
        require(3<=sync_i-raw_i<=5,'restart crosses real synchronizer')
        require(all(x['synced']=='1' for x in life[sync_i:]),'retained synchronized restart')
    else:require(s['idle'] and s['drained'] and not l['banks'] and not l['pending'],'healthy pool drained after four jobs')
    return dict(case=case_id,cycles=len(raw),counts=dict(counts),control=ctl,metrics=metrics,
                memory_events=len(mem),retained=bool(case_id))

def review(text):
    require(not re.search(r'^(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    require(text.splitlines().count('NTRD_LIFETIME_SUITE_PASS cases=8')==1,'one suite finish')
    results=[]
    for case in range(8):
        prefix=f'NTRD case={case} '
        subset='\n'.join(x[len(prefix):] for x in text.splitlines() if x.startswith(prefix))
        require(subset, 'case trace exists');results.append(review_case(subset,case))
    require(sum(x.startswith('NTRD case=') and ' CASE_PASS ' in x for x in text.splitlines())==8,'all cases once')
    return dict(status='PASS_NTRD_REAL_POOL_CDC_RETAINED_LIFETIME',cases=results,
                scope='Actual NTRD/collector/bridge/supervisor/pool/AMD CDC; memory clocks/calibration and AXI destinations are protocol drivers. Seven runtime-fault cases preserve cold ownership/control/AXI obligations; one four-job recovery scenario. No actual MIG, physical reconfiguration, metastability, routed timing or inference qualification.')
