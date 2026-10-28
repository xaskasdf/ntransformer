"""Transaction, finite-prefix, payload and response-slot audit of active AXI4."""
import collections,json,re,sys
from check_ntrpu_axi4_response_guard import pattern,require

ADDRESS=['addr','id','len','size','prot','burst','cache','lock','region','qos']
PAYLOAD={'aw':['aw'+x for x in ADDRESS],'ar':['ar'+x for x in ADDRESS],
         'w':['wdata','wstrb','wlast'],'b':['bid','bresp'],'r':['rid','rresp','rdata','rlast']}
FAULTS=[0,0,0,0,0,2,4,32,16,3,0,4,32]

class Credits:
    """Unbounded Python queues, with reservations for protected output slots."""
    def __init__(self):
        self.aw=collections.deque();self.w=collections.deque();self.bc=collections.Counter()
        self.rq=[collections.deque() for _ in range(16)];self.rp=[0]*16
        self.ap=self.wp=self.partial=self.fault=0
        self.bslot=self.rslot=None;self.bs=self.rs=False;self.oldb=self.oldr=None

    def idle(self,s):
        return self.bslot is None and self.rslot is None and self.ap==self.wp==self.partial==sum(self.rp)==self.fault==0 and not(s['m_bvalid'] or s['m_rvalid'])

    def observe(self,s,forward,context):
        p=lambda channel:tuple(s['m_'+x] for x in PAYLOAD[channel])
        af=s['m_awvalid'] and s['m_awready'];wf=s['m_wvalid'] and s['m_wready'];arf=s['m_arvalid'] and s['m_arready']
        hb=forward and s['s_bvalid'] and s['s_bready'];hr=forward and s['s_rvalid'] and s['s_rready'];wc=wf and s['m_wlast']
        if forward:
            require(bool(s['s_bvalid'])==(self.bslot is not None) and bool(s['s_rvalid'])==(self.rslot is not None),context+' protected slots')
            if self.bslot is not None:require(tuple(s['s_'+x] for x in PAYLOAD['b'])==self.bslot,context+' protected B payload')
            if self.rslot is not None:require(tuple(s['s_'+x] for x in PAYLOAD['r'])==self.rslot,context+' protected R512 payload')
        else:require(self.bslot is None and self.rslot is None,context+' no hidden forwarded replies in REJECT')
        pair=bool(self.aw and self.w);good=pair and self.aw[0][1]==self.w[0];pid=self.aw[0][0] if self.aw else -1
        ba=self.bc[s['m_bid']]>0 or (good and pid==s['m_bid'])
        ra=bool(self.rq[s['m_rid']]);rl=ra and self.rq[s['m_rid']][0]==1
        bp=p('b');rp=p('r');bc=self.bs and (not s['m_bvalid'] or bp!=self.oldb);rc=self.rs and (not s['m_rvalid'] or rp!=self.oldr)
        partial_bad=len(self.aw)>len(self.w) and self.partial>0 and self.partial>=self.aw[len(self.w)][1]
        wb=(af and self.ap==32 and not hb) or (wf and self.partial==0 and self.wp==32 and not hb) or (pair and not good) or partial_bad or (wf and (self.partial==256 or (self.partial==255 and not s['m_wlast'])))
        rb=arf and self.rp[s['m_arid']]==8 and not(hr and s['s_rlast'] and s['s_rid']==s['m_arid'])
        bb=(s['m_bvalid'] and not ba) or bc;rrb=(s['m_rvalid'] and (not ra or bool(s['m_rlast'])!=rl)) or rc
        br=(self.bslot is None or (forward and s['s_bready'])) and ba and not(self.fault&5) and not wb and not bc
        rr=(self.rslot is None or (forward and s['s_rready'])) and ra and bool(s['m_rlast'])==rl and not(self.fault&10) and not rb and not rc
        require(bool(s['m_bready'])==bool(br) and bool(s['m_rready'])==bool(rr),context+' return permission')
        cb=s['m_bvalid'] and br;cr=s['m_rvalid'] and rr;crlast=cr and rl
        if not(self.fault&4) and not wb:
            self.ap+=int(af)-int(hb);self.wp+=int(wc)-int(hb)
            if good:self.aw.popleft();self.w.popleft();self.bc[pid]+=1
            if cb:self.bc[s['m_bid']]-=1
            if af:self.aw.append((s['m_awid'],s['m_awlen']+1))
            if wf:
                self.partial+=1
                if wc:self.w.append(self.partial);self.partial=0
        if not(self.fault&8) and not rb:
            if cr:
                self.rq[s['m_rid']][0]-=1
                if crlast:self.rq[s['m_rid']].popleft()
            if arf:self.rq[s['m_arid']].append(s['m_arlen']+1);self.rp[s['m_arid']]+=1
            if hr and s['s_rlast']:self.rp[s['s_rid']]-=1
        if hb:self.bslot=None
        if hr:self.rslot=None
        if cb:self.bslot=bp
        if cr:self.rslot=rp
        self.fault|=int(bool(bb))|(int(bool(rrb))<<1)|(int(bool(wb))<<2)|(int(bool(rb))<<3)
        self.bs=bool(s['m_bvalid'] and not br);self.rs=bool(s['m_rvalid'] and not rr);self.oldb=bp;self.oldr=rp

def review(text):
    require(not re.search(r'^\s*(Fatal:|ERROR:|Error:|Failure:|Warning:|WARNING:)',text,re.M),'runtime diagnostic')
    require(text.splitlines().count('AXI4_ADMISSION_SUITE_PASS cases=13 write_depth=32 read_total=128 native_endpoint=0')==1,'complete scoped suite')
    require(re.findall(r'^AXI4_ADMISSION_START case=(\d+)$',text,re.M)==[str(i) for i in range(13)],'ordered starts')
    result=[]
    for case in range(13):
        a=text.index(f'AXI4_ADMISSION_START case={case}\n');z=text.index(f'AXI4_ADMISSION_START case={case+1}\n') if case<12 else text.index('AXI4_ADMISSION_SUITE_PASS')
        section=text[a:z];samples=[]
        for line in section.splitlines():
            if line.startswith('AXI4_ADMISSION_SAMPLE '):
                row=dict(x.split('=') for x in line.split()[1:]);samples.append({k:int(v,10 if k in ['case','cycle','time'] else 16) for k,v in row.items()})
        require(len(samples)>10,'samples present')
        require(re.findall(r'^AXI4_ADMISSION_CASE_PASS case=(\d+) faults=(\d+) cycles=(\d+)$',section,re.M)==[(str(case),str(FAULTS[case]),str(len(samples)))],'terminal case and complete samples')
        guard=Credits();state=0;request_fault=local_fault=False
        retained={k:None for k in ['aw','w','ar']};held={k:False for k in retained}
        targets=None;aw_left=w_left=0;ar_left=False
        captured=False;tag=quiet=0
        maws=[];mws=[];mars=[];mwpartial=[];uaws=[];uars=[];uws=[];uwpartial=[]
        lb_aw=None;lb_wdone=False;lb_partial=0;lb_valid=False;lr_desc=None
        counts=collections.Counter();maxima=[0,0,0];proof_tags=set();local_held_proofs=0;partial_reserved=False
        m_read=[collections.deque() for _ in range(16)];m_write=collections.Counter();pair_index=0
        previous_time=None;last_stalled={};last_payload={};prefix_log=None
        for cyc,s in enumerate(samples):
            ctx=f'case{case}/cycle{cyc}'
            require(s['cycle']==cyc and s['case']==case and s['cold_rst_n']==1,ctx+' cycle identity')
            if previous_time is not None:require(s['time']-previous_time==10000,ctx+' uninterrupted clock')
            previous_time=s['time']
            flags=guard.fault | (int(request_fault)<<4) | (int(local_fault)<<5)
            require(s['state_debug']==state and s['fault_sources']==flags and bool(s['protocol_fault'])==bool(flags),ctx+' state and retained faults')
            require((s['aw_pending'],s['w_pending'],s['ar_pending'],s['w_partial'])==(guard.ap,guard.wp,sum(guard.rp),guard.partial),ctx+' guarded inventory')
            require((s['aw_left'],s['w_left'],bool(s['ar_left']))==(aw_left,w_left,ar_left),ctx+' finite remaining prefix')
            maxima=[max(maxima[0],guard.ap),max(maxima[1],guard.wp),max(maxima[2],sum(guard.rp))]
            raw={ch:tuple(s['s_'+x] for x in PAYLOAD[ch]) for ch in retained}
            up={ch:retained[ch] if retained[ch] is not None else raw[ch] for ch in retained}
            uv={ch:bool(s['s_'+ch+'valid'] or retained[ch] is not None) for ch in retained}
            request_bad=any(retained[ch] is not None and (not s['s_'+ch+'valid'] or raw[ch]!=retained[ch]) for ch in retained)
            cap={'aw':not(guard.fault&4) and guard.ap<32,'w':not(guard.fault&4) and (guard.partial>0 or guard.wp<32),'ar':not(guard.fault&8) and guard.rp[s['m_arid']]<8}
            remaining={'aw':aw_left,'w':w_left,'ar':ar_left}
            fire={};ufire={}
            for ch in retained:
                permit=held[ch] or (cap[ch] and (state==0 or remaining[ch]>0))
                expected_mvalid=state<2 and uv[ch] and permit
                require(bool(s['m_'+ch+'valid'])==bool(expected_mvalid),ctx+' request admission '+ch)
                if expected_mvalid:require(tuple(s['m_'+x] for x in PAYLOAD[ch])==up[ch],ctx+' preserved request payload '+ch)
                ready=((lb_aw is None and not lb_valid and not local_fault) if ch=='aw' else (not lb_wdone and not lb_valid and not local_fault) if ch=='w' else lr_desc is None) if state==2 else permit and s['m_'+ch+'ready']
                require(bool(s['s_'+ch+'ready'])==bool(ready),ctx+' upstream readiness '+ch)
                fire[ch]=bool(s['m_'+ch+'valid'] and s['m_'+ch+'ready']);ufire[ch]=bool(uv[ch] and ready)
                if state<2:require(fire[ch]==ufire[ch],ctx+' forwarded handshake')
            fence=state!=0 and aw_left==w_left==0 and not ar_left and not any(held.values()) and not any(s['m_'+ch+'valid'] for ch in retained)
            require(bool(s['source_fenced'])==fence,ctx+' source fence')
            gidle=guard.idle(s)
            ack=bool(s['proof_request'] and captured and tag==s['proof_tag'] and state==2 and gidle and not flags and quiet==2)
            require(bool(s['drain_ack'])==ack and s['drain_tag']==tag,ctx+' fresh section certificate')
            lidle=state==2 and not flags and lb_aw is None and not lb_wdone and lb_partial==0 and not lb_valid and lr_desc is None and not any(uv.values())
            require(bool(s['local_idle'])==lidle,ctx+' distinct local inventory')
            if ack:
                proof_tags.add(s['proof_tag'])
                local_held_proofs+=int(bool(s['s_bvalid'] or s['s_rvalid']))
            # Every protected output and downstream request must remain stable
            # across stalls, even during a fault or a state transition.
            for side,ch in [('m','aw'),('m','w'),('m','ar'),('s','b'),('s','r')]:
                key=side+ch;valid=s[side+'_'+ch+'valid'];ready=s[side+'_'+ch+'ready'];payload=tuple(s[side+'_'+x] for x in PAYLOAD[ch])
                if last_stalled.get(key):require(valid and payload==last_payload[key],ctx+' stable offered '+key)
                last_stalled[key]=bool(valid and not ready);last_payload[key]=payload
            if state==0 and (s['close_request'] or flags or request_bad):
                # Absolute accepted transaction histories, not RTL ring pointers
                # or pending counters, define the finite group on a healthy close.
                target_w=max(len(maws)+int(uv['aw']),len(mws)+int(bool(mwpartial) or uv['w']))
                targets=(target_w,len(mars)+int(uv['ar']))
                prefix_log=dict(aw_target=target_w,ar_target=targets[1],aw_accepted=len(maws),w_completed=len(mws),w_partial=len(mwpartial))
                partial_reserved=bool(mwpartial)
            if targets is not None:
                require(len(maws)<=targets[0] and len(mws)<=targets[0] and len(mars)<=targets[1],ctx+' bounded forwarded histories')
                if state==1:
                    if s['m_awvalid']:require(len(maws)<targets[0],ctx+' no AW beyond prefix')
                    if s['m_wvalid']:require(len(mws)<targets[0],ctx+' no W beyond prefix')
                    if s['m_arvalid']:require(len(mars)<targets[1],ctx+' no AR beyond prefix')
            if state==2:
                require(not any(s['m_'+ch+'valid'] for ch in retained),ctx+' permanently closed')
                require(bool(s['s_bvalid'])==lb_valid and bool(s['s_rvalid'])==(lr_desc is not None),ctx+' local response validity')
                if lb_valid:require((s['s_bid'],s['s_bresp'])==(lb_aw[1],2),ctx+' local BID/SLVERR')
                if lr_desc is not None:require((s['s_rid'],s['s_rresp'],s['s_rdata'],s['s_rlast'])==(lr_desc[0],2,0,int(lr_desc[1]==1)),ctx+' local RID/LEN/RLAST/SLVERR')
            # Independent downstream responder oracle. Inputs with no credit are
            # actual injected faults; captured responses must have causal credit.
            cb=bool(s['m_bvalid'] and s['m_bready']);cr=bool(s['m_rvalid'] and s['m_rready'])
            while pair_index<min(len(maws),len(mws)):
                ad=maws[pair_index];burst=mws[pair_index]
                if len(burst)==ad[2]+1:m_write[ad[1]]+=1
                pair_index+=1
            if cb:
                require(m_write[s['m_bid']]>0 and s['m_bresp']==s['m_bid']%4,ctx+' fixture B credit/response');m_write[s['m_bid']]-=1
            if cr:
                require(bool(m_read[s['m_rid']]),ctx+' fixture R credit')
                desc=m_read[s['m_rid']][0];addr,beats,beat=desc
                seed=((addr&0xffffffff)>>12)^(addr>>32)^(beat<<8)^s['m_rid']
                require((s['m_rdata'],s['m_rresp'],s['m_rlast'])==(pattern(seed),(beat+s['m_rid'])%4,int(beat+1==beats)),ctx+' independent R512 fixture')
                desc[2]+=1
                if desc[2]==beats:m_read[s['m_rid']].popleft()
            # Guard advances against the pre-edge return slots and raw requests.
            guard.observe(s,state<2,ctx)
            old_state=state
            if state==0 and (s['close_request'] or flags or request_bad):
                state=1
                aw_left=targets[0]-len(maws)-int(fire['aw'])
                w_left=targets[0]-len(mws)-int(fire['w'] and s['m_wlast'])
                ar_left=uv['ar'] and not fire['ar']
            elif state==1:
                aw_left-=int(fire['aw']);w_left-=int(fire['w'] and s['m_wlast'])
                if fire['ar']:ar_left=False
                if fence and gidle and not flags:state=2
            elif state==2:
                if lb_aw is not None and lb_wdone and not lb_valid:
                    if lb_aw[2]+1==lb_partial:lb_valid=True
                    else:local_fault=True
                if s['s_bvalid'] and s['s_bready']:lb_aw=None;lb_wdone=False;lb_partial=0;lb_valid=False
                if s['s_rvalid'] and s['s_rready']:
                    lr_desc[1]-=1
                    if lr_desc[1]==0:lr_desc=None
                if ufire['aw']:lb_aw=up['aw']
                if ufire['w']:
                    if lb_partial==255 and not up['w'][2]:local_fault=True
                    else:lb_partial+=1;lb_wdone=bool(up['w'][2])
                if ufire['ar']:lr_desc=[up['ar'][1],up['ar'][2]+1]
            request_fault|=request_bad
            if not s['proof_request']:captured=False;quiet=0
            elif not captured or tag!=s['proof_tag']:captured=True;tag=s['proof_tag'];quiet=0
            elif old_state!=2 or not gidle or flags:quiet=0
            else:quiet=min(2,quiet+1)
            for ch in retained:
                if uv[ch] and not s['s_'+ch+'ready']:
                    if retained[ch] is None:retained[ch]=raw[ch]
                else:retained[ch]=None
                held[ch]=bool(s['m_'+ch+'valid'] and not s['m_'+ch+'ready'])
                counts[('local_' if old_state==2 else 'forward_')+ch]+=int(ufire[ch])
            if ufire['aw']:uaws.append(up['aw'])
            if ufire['ar']:uars.append(up['ar'])
            if ufire['w']:
                uwpartial.append(up['w'])
                if up['w'][2]:uws.append(uwpartial);uwpartial=[]
            if fire['aw']:maws.append(up['aw'])
            if fire['w']:
                mwpartial.append(up['w'])
                if up['w'][2]:mws.append(mwpartial);mwpartial=[]
            if fire['ar']:
                mars.append(up['ar']);m_read[up['ar'][1]].append([up['ar'][0],up['ar'][2]+1,0])
            counts['raw_b']+=int(cb);counts['raw_r']+=int(cr)
            counts[('local_' if old_state==2 else 'forward_')+'b']+=int(bool(s['s_bvalid'] and s['s_bready']))
            counts[('local_' if old_state==2 else 'forward_')+'r']+=int(bool(s['s_rvalid'] and s['s_rready']))
        # Source stimulus is checked as well as transparency, including all
        # address bits/attributes and W data/strobes buffered before AW exists.
        for ad in uaws:
            addr,id_,len_,size,prot,burst,cache,lock,region,qos=ad;n=(addr&0xffffffff)>>12
            require(addr==0x100000000+4096*n and (id_,size,prot,burst,cache,lock,region,qos)==(n%16,4 if len_==255 else 6,n%8,1,3+4*(n%4),0,n%16,15-n%16),'AW fixture full attributes')
        for ad in uars:
            addr,id_,len_,size,prot,burst,cache,lock,region,qos=ad;n=(addr&0xffffffff)>>12
            require(addr==0x200000000+4096*n and (id_,size,prot,burst,cache,lock,region,qos)==(n%16,4 if len_==255 else 6,n%8,1,3+4*(n%4),0,n%16,15-n%16),'AR fixture full attributes')
        for idx,burst in enumerate(uws):
            require(idx<len(uaws) or (case==11 and idx==1),'complete W has AW by end')
            ad=uaws[idx] if idx<len(uaws) else (0x100018000,8,1)
            n=(ad[0]&0xffffffff)>>12
            intentional_mismatch=(case==6 and idx==1) or (case==7 and idx==0) or (case==11 and idx==0)
            require(len(burst)==ad[2]+1 or intentional_mismatch,'write LEN/WLAST pairing')
            for beat,(data,strb,last) in enumerate(burst):
                mask=(0xffff<<((beat%4)*16)) if ad[2]==255 else (0xaa55aa55aa55aa55^(0x0101010101010101*beat))&((1<<64)-1)
                require((data,strb,last)==(pattern(4096*n+beat),mask,int(beat==len(burst)-1)),'W512 fixture payload/strobe/LAST')

        if case==10:
            require(prefix_log['w_partial']==3 and targets==(1,0) and ack and lidle,'partial W gap reaches fresh proof')
            close_row=next(row for row in samples if row['state_debug']==0 and row['close_request'])
            require(close_row['s_wvalid']==0 and close_row['w_partial']==3,'actual gap at close')
        if case==11:
            require(any(row['fault_sources']==4 and row['m_wvalid'] and not row['m_wready'] for row in samples),'held W during capacity loss')
            require(len(mws)==2 and len(maws)==1 and guard.partial==1,'accepted final held W with frozen corrupt inventory')
        if case==12:
            require(len(uwpartial)==256 and not uws and len(uaws)==1,'missing local WLAST fixture')
            for beat,(data,strb,last) in enumerate(uwpartial):
                require((data,strb,last)==(pattern(4096*26+beat),0xffff<<((beat%4)*16),0),'partial W512 fixture')
            require(lb_partial==255 and counts['local_r']==4 and counts['local_b']==0,'bounded local partial inventory and healthy R')
        require((guard.fault|(int(request_fault)<<4)|(int(local_fault)<<5))==FAULTS[case],'final faults')
        if case<5:
            require(state==2 and targets==(len(maws),len(mars)) and len(mws)==len(maws),'entire finite group completed')
            require(counts['forward_aw']==counts['forward_b'] and (counts['forward_ar']>0 or case==2),'healthy forwarded completion')
            require(ack and lidle,'healthy terminal proof/local idle')
        if case==0:require(maxima==[32,32,128] and targets==(33,129),'full capacity plus blocked extra requests')
        if case==1:require(prefix_log['w_completed']==31 and prefix_log['w_partial']==3 and targets==(32,1),'W-before-AW prefix')
        if case==2:require(prefix_log['aw_accepted']==32 and targets==(33,0),'AW-before-W prefix')
        if case==3:require(len(proof_tags)==2 and local_held_proofs>5 and targets==(1,1),'fresh proof while local replies held')
        if case==4:require(partial_reserved and targets==(1,1),'partial W prefix')
        if FAULTS[case]:require(not ack,'fault never certifies')
        result.append(dict(case=case,cycles=len(samples),faults=FAULTS[case],max_aw=maxima[0],max_w=maxima[1],max_ar=maxima[2],
            prefix=prefix_log,transactions=dict(counts),proof_tags=sorted(proof_tags),local_reply_cycles_with_section_proof=local_held_proofs))
    return dict(status='PASS_THIRTEEN_RETAINED_AXI4_ADMISSION_SCENARIOS',cases=result,qualified_native_endpoint=False,qualified_physical_design=False)

if __name__=='__main__':print(json.dumps(review(open(sys.argv[1],encoding='utf-8').read()),indent=2))
