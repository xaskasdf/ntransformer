"""Independent deque/response-slot reconstruction of the causal AXI4 guard."""
import collections,json,re,sys

def require(ok,message):
    if not ok:raise ValueError(message)

def pattern(seed):
    return sum((((0x9e3779b9*(seed+j+1))^0xd00dbeef)&0xffffffff)<<(32*j) for j in range(16))

def length(n):return [1,2,4,16,64,256,3,1][n%8]

FAULTS=[0,0,0,0,1,1,1,2,2,2,2,4,8,4,4,1,2,1,2]

def review(text):
    require(not re.search(r'^\s*(Fatal:|ERROR:|Error:|Failure:|WARNING:|Warning:)',text,re.M),'runtime diagnostic')
    require(text.splitlines().count('AXI4_GUARD_SUITE_PASS cases=19 write_depth=32 read_total=128 native_endpoint=0')==1,'complete scoped suite')
    starts=[int(x) for x in re.findall(r'^AXI4_GUARD_START case=(\d+)$',text,re.M)]
    require(starts==list(range(19)),'ordered case starts')
    results=[]
    for case in range(19):
        start=text.index(f'AXI4_GUARD_START case={case}\n')
        end=text.index(f'AXI4_GUARD_START case={case+1}\n') if case<18 else text.index('AXI4_GUARD_SUITE_PASS')
        section=text[start:end]
        samples=[]
        for line in section.splitlines():
            if not line.startswith('AXI4_GUARD_SAMPLE '):continue
            fields=dict(item.split('=') for item in line.split()[1:])
            require(int(fields['case'])==case,'case boundary')
            samples.append({k:int(v,10 if k in ['case','cycle','time'] else 16) for k,v in fields.items()})
        require(len(samples)>5,'samples present')
        pass_rows=re.findall(r'^AXI4_GUARD_CASE_PASS case=(\d+) faults=(\d+) cycles=(\d+)$',section,re.M)
        require(pass_rows==[(str(case),str(FAULTS[case]),str(len(samples)))],'case pass and all sampled cycles')
        aw=collections.deque();w=collections.deque();bc=collections.Counter()
        rq=[collections.deque() for _ in range(16)];rp=[0]*16
        ap=wp=partial=fault=0;bs=rs=False;oldb=oldr=None;bslot=rslot=None
        tx=collections.Counter();aw_seen=[];ar_seen=[];w_lengths=[];observed_partial=0
        cb_seen=[];cr_seen=[];hb_seen=[];hr_seen=[];max_ap=max_wp=max_rp=0
        stalled_b=stalled_r=0;previous_time=None
        for cyc,s in enumerate(samples):
            context=f'case{case}/cycle{cyc}'
            require(s['cycle']==cyc and s['cold_rst_n']==1,context+' complete cycle order')
            if previous_time is not None:require(s['time']-previous_time==10000,context+' clock interval')
            previous_time=s['time']
            require((s['aw_pending'],s['w_pending'],s['ar_pending'],s['w_partial'],s['fault_sources'])==(ap,wp,sum(rp),partial,fault),context+' retained inventory')
            require(bool(s['h_bvalid'])==(bslot is not None) and bool(s['h_rvalid'])==(rslot is not None),context+' output slots')
            if bslot is not None:require((s['h_bid'],s['h_bresp'])==bslot,context+' stable protected B')
            if rslot is not None:require((s['h_rid'],s['h_rresp'],s['h_rdata'],s['h_rlast'])==rslot,context+' stable protected R512')
            require((s['aw_capacity'],s['w_capacity'],s['ar_capacity'])==(int(not(fault&4) and ap<32),int(not(fault&4) and (partial>0 or wp<32)),int(not(fault&8) and rp[s['arid']]<8)),context+' admission capacity')
            idle=bslot is None and rslot is None and ap==wp==partial==sum(rp)==0 and fault==0 and not(s['c_bvalid'] or s['c_rvalid'])
            require(bool(s['local_idle'])==idle,context+' local idle')
            af=s['awvalid'] and s['awready'];wf=s['wvalid'] and s['wready'];arf=s['arvalid'] and s['arready']
            hb=s['h_bvalid'] and s['h_bready'];hr=s['h_rvalid'] and s['h_rready']
            wc=wf and s['wlast'];hrlast=hr and s['h_rlast']
            pair=bool(aw and w);good=pair and aw[0][1]==w[0];pid=aw[0][0] if aw else -1
            bavailable=bc[s['c_bid']]>0 or (good and pid==s['c_bid'])
            ravailable=bool(rq[s['c_rid']]);rfinal=ravailable and rq[s['c_rid']][0]==1
            bp=(s['c_bid'],s['c_bresp']);rpayload=(s['c_rid'],s['c_rresp'],s['c_rdata'],s['c_rlast'])
            bchange=bs and (not s['c_bvalid'] or bp!=oldb)
            rchange=rs and (not s['c_rvalid'] or rpayload!=oldr)
            # Python unbounded queues/counters have no RTL circular indices.
            partial_bad=len(aw)>len(w) and partial>0 and partial>=aw[len(w)][1]
            wb=(af and ap==32 and not hb) or (wf and partial==0 and wp==32 and not hb) or (pair and not good) or partial_bad or (wf and (partial==256 or (partial==255 and not s['wlast'])))
            rb=arf and rp[s['arid']]==8 and not(hrlast and s['h_rid']==s['arid'])
            bbad=(s['c_bvalid'] and not bavailable) or bchange
            rbad=(s['c_rvalid'] and (not ravailable or bool(s['c_rlast'])!=rfinal)) or rchange
            br=(bslot is None or s['h_bready']) and bavailable and not(fault&5) and not wb and not bchange
            rr=(rslot is None or s['h_rready']) and ravailable and bool(s['c_rlast'])==rfinal and not(fault&10) and not rb and not rchange
            require(bool(s['c_bready'])==bool(br) and bool(s['c_rready'])==bool(rr),context+' causal capture permission')
            cb=s['c_bvalid'] and br;cr=s['c_rvalid'] and rr;crlast=cr and rfinal
            for key,value in [('aw',af),('w',wf),('wlast',wc),('ar',arf),('cb',cb),('cr',cr),('hb',hb),('hr',hr)]:tx[key]+=int(value)
            stalled_b+=int(s['h_bvalid'] and not s['h_bready']);stalled_r+=int(s['h_rvalid'] and not s['h_rready'])
            if af:aw_seen.append((s['awid'],s['awlen']+1))
            if arf:ar_seen.append((s['arid'],s['arlen']+1))
            if wf:
                observed_partial+=1
                if wc:w_lengths.append(observed_partial);observed_partial=0
            if cb:cb_seen.append(bp)
            if cr:cr_seen.append(rpayload)
            if hb:hb_seen.append(bslot)
            if hr:hr_seen.append(rslot)
            if not(fault&4) and not wb:
                ap+=int(af)-int(hb);wp+=int(wc)-int(hb)
                if good:aw.popleft();w.popleft();bc[pid]+=1
                if cb:bc[s['c_bid']]-=1
                if af:aw.append((s['awid'],s['awlen']+1))
                if wf:
                    partial+=1
                    if wc:w.append(partial);partial=0
            if not(fault&8) and not rb:
                if cr:
                    rq[s['c_rid']][0]-=1
                    if crlast:rq[s['c_rid']].popleft()
                if arf:rq[s['arid']].append(s['arlen']+1);rp[s['arid']]+=1
                if hrlast:rp[s['h_rid']]-=1
            if hb:bslot=None
            if hr:rslot=None
            if cb:bslot=bp
            if cr:rslot=rpayload
            fault|=int(bbad)|(int(rbad)<<1)|(int(bool(wb))<<2)|(int(bool(rb))<<3)
            bs=bool(s['c_bvalid'] and not br);rs=bool(s['c_rvalid'] and not rr);oldb=bp;oldr=rpayload
            max_ap=max(max_ap,ap);max_wp=max(max_wp,wp);max_rp=max(max_rp,sum(rp))
        require(fault==FAULTS[case],'final exact fault')
        require(hb_seen==cb_seen[:len(hb_seen)] and hr_seen==cr_seen[:len(hr_seen)],'no fabricated/reordered protected response')
        if case==0:
            require(aw_seen==[(i%16,length(i)) for _ in range(3) for i in range(32)],'write fixture addresses/lengths')
            require(w_lengths==[length(i) for _ in range(3) for i in range(32)],'write fixture WLASTs')
            require(cb_seen==[(i,resp) for k in range(3) for i in range(15,-1,-1) for resp in [(i+k)%4,(i+k+1)%4]],'B reordering across IDs')
            require(max_ap==max_wp==32 and tx['aw']==tx['hb']==96,'write capacity and wraps')
        elif case==1:
            expected=[length(i) for i in range(31)]+[4]
            require(aw_seen==[(i%16,n) for i,n in enumerate(expected)] and w_lengths==expected,'W-before-AW mixed bursts')
            first_aw=next(i for i,s in enumerate(samples) if s['awvalid'])
            require(samples[first_aw]['w_pending']==31 and samples[first_aw]['w_partial']==3,'31 completed plus partial W before AW')
            require(cb_seen==[(i,resp) for i in range(15,-1,-1) for resp in [0,2]],'W-before-AW B identity')
        elif case==2:
            require(ar_seen==[(i,length(j)) for i in range(16) for j in range(8)],'all read descriptors')
            expected=[(i,seed%4,pattern(seed),int(beat==length(j)-1)) for j in range(8) for beat in range(length(j)) for i in range(16) for seed in [4096*j+256*beat+i]]
            require(cr_seen==expected and hr_seen==expected and max_rp==128,'interleaved full512 read payloads and all128 capacity')
        elif case==3:
            require(aw_seen==ar_seen==[(i%16,1) for i in range(96)],'duplex requests')
            require(cb_seen==[((i-1)%16,i%4) for i in range(1,97)],'duplex B')
            require(cr_seen==[((i-1)%16,i%4,pattern(i),1) for i in range(1,97)],'duplex R')
            require(sum(all(s[k] for k in ['awvalid','wvalid','arvalid','c_bvalid','c_bready','c_rvalid','c_rready','h_bvalid','h_rvalid']) for s in samples)>90,'simultaneous pipeline')
        if case<4:
            require(ap==wp==sum(rp)==partial==fault==0 and bslot is None and rslot is None,'fully drained positive case')
        else:
            expected_cb={5:[(1,0)],7:[(1,0)],9:[(2,0)],11:[(0,0)],12:[(2,0)],15:[(0,0)],17:[(0,0)]}.get(case,[])
            expected_cr={6:[(2,2,pattern(2),1)],8:[(1,1,pattern(1),0)],12:[(0,0,pattern(0),1)],16:[(0,0,pattern(0),1)],18:[(0,0,pattern(0),1)]}.get(case,[])
            require(cb_seen==expected_cb and cr_seen==expected_cr,'exact permitted captures during faults')
            require(hb_seen==(expected_cb if case!=17 else []) and hr_seen==(expected_cr if case!=18 else []),'healthy output preservation/completion')
        results.append(dict(case=case,cycles=len(samples),faults=fault,transactions=dict(tx),max_aw=max_ap,max_w=max_wp,max_ar=max_rp,protected_b_stalls=stalled_b,protected_r_stalls=stalled_r))
    return dict(status='PASS_NINETEEN_AXI4_RESPONSE_GUARD_SCENARIOS',cases=results,qualified_native_endpoint=False)

if __name__=='__main__':print(json.dumps(review(open(sys.argv[1],encoding='utf-8').read()),indent=2))
