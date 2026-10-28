"""Automatic guard integration; derived from the frozen boundary inventory checker."""
import collections,re

REQ={0,1,3,4,6};RESP={2,5,7};WIDTHS=[82,577,6,32,36,2,32,34]
def require(ok,message):
    if not ok:raise ValueError(message)
def fields(line):return dict(re.findall(r'(\w+)=([^\s]+)',line))
def num(value,base=16):
    require(re.fullmatch('[0-9a-fA-F]+',value) is not None,'unknown active value')
    return int(value,base)
def value(c,n):
    if c==0:return ((0x1234000100001000+64*n)<<18)|((n&15)<<14)|(6<<3)|2
    if c==1:return sum(((n*37+b*19)&255)<<(65+8*b) for b in range(64))|((2**64-1)<<1)|1
    if c==2:return ((n&15)<<2)|(2 if n%2 else 0)
    if c==3:return 0x20+4*n
    if c==4:return ((0x87654320+n)<<4)|15
    if c==5:return 3 if n%2 else 0
    if c==6:return 0x80+4*n
    if c==7:return ((0x5a1c0000+n)<<2)|(3 if n%2 else 0)

def review(text,complete=True,expected_cases=17):
    if complete:require(not re.search(r'(?mi)^\s*(?:Fatal:|ERROR:|FAIL )',text),'simulation failure')
    groups=collections.defaultdict(lambda:collections.defaultdict(list));payload={}
    for line in text.splitlines():
        kind=line.split(' ',1)[0]
        if not kind.startswith('BOUND_'):continue
        f=fields(line)
        if kind in {'BOUND_CUT_DATA','BOUND_HOST_DATA'}:
            key=(kind,int(f['case']),int(f['cycle']),int(f['c']))
            require(key not in payload,'duplicate payload');payload[key]=f
        elif 'case' in f:groups[int(f['case'])][kind].append(f)
    require(groups and sorted(groups)==list(range(len(groups))),'continuous case inventory')
    results=[];used=set()
    for scenario,g in sorted(groups.items()):
        rows=g['BOUND_CUT'];host=g['BOUND_HOST']
        require(rows and host,'both domains witnessed')
        q=[collections.deque() for _ in range(8)]
        rx=[[] for _ in range(8)];tx=[[] for _ in range(8)];dropped=[[] for _ in range(8)]
        offer=[[] for _ in range(8)];host_rx=[[] for _ in range(8)];host_tx=[[] for _ in range(8)]
        held_in={};held_out={};previous=None;seen_invalid=False;fenced=0;max_drop=0
        for row in rows:
            cycle=int(row['cycle']);ps=int(row['ps']);bound=num(row['bound']);invalid=num(row['invalid'])
            sv,sr,mv,mr=[num(row[k]) for k in ['sv','sr','mv','mr']]
            require(num(row['cold'])==1 and num(row['fault'])==0,'retained lifetime and no contract fault')
            require(num(row['fenced'])==fenced,'reset fence is retained')
            require(not seen_invalid or invalid==1,'loss cannot be forgotten');seen_invalid|=bool(invalid)
            if previous:
                dt=ps-int(previous['ps'])
                require(cycle==int(previous['cycle'])+1 and (dt==5000 or (scenario==15 and dt>5000 and dt%2500==0)),'source clock cadence or explicit stopped-source case')
            for c in range(8):
                v,r,ov,ore=[(x>>c)&1 for x in [sv,sr,mv,mr]]
                pending=(num(row['pending'])>>(2*c))&3
                require(pending==len(q[c])<=2,'retained pending inventory')
                entry=None
                if v or ov:
                    key=('BOUND_CUT_DATA',scenario,cycle,c);require(key in payload,'missing source payload');used.add(key);entry=payload[key]
                # Bench arrays pad every channel to577bits. Only the declared
                # port width is connected to the DUT (not sign-extended padding).
                inp=(num(entry['input']) & ((1<<WIDTHS[c])-1)) if v else None
                out=num(entry['output']) if ov else None
                if c in held_in and (c in RESP or (previous['bound']=='1' and bound)):
                    require(v and inp==held_in[c],'producer held payload stable')
                if c in held_out and (c in REQ or (previous['bound']=='1' and bound)):
                    require(ov and out==held_out[c],'consumer held payload stable across reset boundary')
                held_in.pop(c,None);held_out.pop(c,None)
                if v and not r:held_in[c]=inp
                if ov and not ore:held_out[c]=out
                discard=c in RESP and (((fenced>>c)&1) or (invalid and not bound))
                if c in REQ and (invalid or not bound):require(not r,'closed request admission')
                if c in RESP and not bound:require(not ov,'facing response interface reset')
                if ov:
                    require(q[c] and out==q[c][0],'ordered immutable output payload '+str((scenario,cycle,c,out,list(q[c]))))
                if ov and ore:tx[c].append(q[c].popleft())
                if v and r:rx[c].append(inp);q[c].append(inp)
                if discard:
                    require(not ov,'no response delivery during abandonment')
                    max_drop=max(max_drop,len(q[c]));dropped[c].extend(q[c]);q[c].clear();fenced|=1<<c
            previous=row
        open_offer={};held_response={};previous=None;clock_gaps=[]
        for row in host:
            cycle=int(row['cycle']);ps=int(row['ps']);up=num(row['rst'])
            hv,hr,ready,valid=[num(row[k]) for k in ['hv','hr','ready','valid']]
            if previous:
                require(cycle==int(previous['cycle'])+1,'continuous host sample indices')
                dt=ps-int(previous['ps']);require(dt>=7300 and dt%3650==0,'host period or stopped-clock gap')
                if dt>7300:clock_gaps.append(dt)
            for c in range(8):
                v,r,ov,ore=[(x>>c)&1 for x in [hv,ready,valid,hr]]
                entry=None
                if v or ov:
                    key=('BOUND_HOST_DATA',scenario,cycle,c);require(key in payload,'missing host payload');used.add(key);entry=payload[key]
                req=num(entry['request']) if v else None;rsp=num(entry['response']) if ov else None
                if not up:open_offer.pop(c,None);held_response.pop(c,None);continue
                if c in REQ and v:
                    if c not in open_offer:offer[c].append(req);open_offer[c]=req
                    require(open_offer[c]==req,'host request stable until acknowledgement')
                    if r:host_tx[c].append(req);open_offer.pop(c)
                if c in held_response:require(ov and rsp==held_response[c],'host response stable until receipt/reset')
                held_response.pop(c,None)
                if c in RESP and ov:
                    if ore:host_rx[c].append(rsp)
                    else:held_response[c]=rsp
            previous=row
        for c in REQ:
            require(rx[c]==offer[c][:len(rx[c])],'source requests originate from ordered host offers')
            require(not dropped[c] and rx[c]==tx[c]+list(q[c]),'requests never discarded on boundary reset')
        for c in RESP:
            require(host_rx[c]==rx[c][:len(host_rx[c])],'host receipts have ordered retained response origin')
        summaries=g['BOUND_SUMMARY']
        case_done=len(g['BOUND_CASE_PASS'])==1
        if case_done:
            require(len(summaries)==8 and {int(x['c']) for x in summaries}==set(range(8)),'complete unique summary')
            for x in summaries:
                c=int(x['c']);require([int(x[k]) for k in ['accepted','delivered','abandoned','pending']]==[len(rx[c]),len(tx[c]),len(dropped[c]),len(q[c])],'independent final counters')
                require(int(x['invalid'])==int(seen_invalid),'final invalidation')
        elif complete:require(False,'unfinished case')
        if complete:
            expected={c:[0,0,0,0] for c in range(8)}
            if scenario==0:expected={c:[3,3,0,0] for c in range(8)};require(not seen_invalid,'normal session remains valid')
            elif 1<=scenario<=5:expected[[0,1,3,4,6][scenario-1]]=[2,2,0,0]
            elif 6<=scenario<=8:
                c=[2,5,7][scenario-6];expected[c]=[3,0,3,0]
                for a in ({2:[0,1],5:[3,4],7:[6]}[c]):expected[a]=[3,3,0,0]
                require(max_drop==3,'two buffered and one new response abandoned together')
            elif 9<=scenario<=11:
                c=[2,5,7][scenario-9];expected[c]=[1,1,0,0] if c==2 else [1,0,1,0]
                for a in ({2:[0,1],5:[3,4],7:[6]}[c]):expected[a]=[1,1,0,0]
                require(len(clock_gaps)==1 and not host_rx[c],'stopped host with response unreceived')
            elif scenario==12:expected[0]=expected[1]=[1,1,0,0];expected[2]=[1,0,1,0]
            elif scenario==13:
                expected[3]=expected[4]=[1,1,0,0]
                require(len(clock_gaps)==1 and not host_tx[3] and not host_tx[4],'destination effects before host request acknowledgements')
            elif scenario==14:
                expected[3]=expected[4]=[1,1,0,0];expected[5]=[1,1,0,0]
                require(len(clock_gaps)==1 and host_rx[5]==[value(5,0)],'host receipt before producer response acknowledgement')
            elif scenario==15:expected[1]=[2,2,0,0]
            elif scenario==16:expected[0]=expected[1]=[1,1,0,0];expected[2]=[2,0,2,0]
            else:require(False,'unreviewed fixture scenario')
            for c in range(8):
                require([len(rx[c]),len(tx[c]),len(dropped[c]),len(q[c])]==expected[c],'scenario obligations '+str((scenario,c)))
                require(rx[c]==[value(c,n) for n in range(len(rx[c]))],'fixture payload bytes '+str((scenario,c)))
            if scenario==0:
                require(all(len(host_tx[c])==3 for c in REQ) and all(len(host_rx[c])==3 for c in RESP),'normal end-to-end host conservation')
            else:require(seen_invalid,'runtime cases remain invalid')
        results.append(dict(case=scenario,completed=case_done,accepted=[len(x) for x in rx],delivered=[len(x) for x in tx],abandoned=[len(x) for x in dropped],pending=[len(x) for x in q],host_request_handshakes=[len(x) for x in host_tx],host_response_handshakes=[len(x) for x in host_rx],clock_gaps_ps=clock_gaps,max_drop_per_edge=max_drop,source_samples=len(rows),host_samples=len(host)))
    require(used==set(payload),'no unpaired payload records')
    if complete:
        require(len(results)==expected_cases,'full case count')
        require(text.splitlines().count('BOUND_SUITE_PASS cases='+str(expected_cases))==1,'one suite marker')
    return dict(status='PASS_RETAINED_SOURCE_BOUNDARY_REAL_CDC' if complete else 'PARTIAL_BOUNDARY_EVIDENCE_NOT_QUALIFIED',cases=results,
                scope='Source-domain retained request/response cut with real AMD AXI4 and AXI-Lite CDCs. Automatic guard with controlled owner/peer acknowledgements and response backpressure shim; no native PCIe, NTRD ownership, DDR or physical qualification. Abandoned counts local unresolved responses, not proof the host did not receive them.')


def review_guard(text):
    result=review(text,expected_cases=17)
    auto=collections.defaultdict(lambda:collections.defaultdict(list))
    cuts=collections.defaultdict(list);hosts=collections.defaultdict(list)
    for line in text.splitlines():
        kind=line.split(' ',1)[0]
        if kind in {'AUTO_MGMT','AUTO_SOURCE','AUTO_EVENT'}:
            f=fields(line);auto[int(f['case'])][kind].append(f)
        elif kind in {'BOUND_CUT','BOUND_HOST'}:
            f=fields(line);(cuts if kind=='BOUND_CUT' else hosts)[int(f['case'])].append(f)
    require(set(auto)==set(range(17)),'guard observations for all cases')
    expected=[0,1,2,1,2,1,2,1,2,4,4,4,2,4,4,8,2]
    checked=[]
    for case,g in sorted(auto.items()):
        m=g['AUTO_MGMT'];s=g['AUTO_SOURCE'];events={x['name']:int(x['ps']) for x in g['AUTO_EVENT']}
        require(m and s,'management/source observations')
        require(len(s)==len(cuts[case]),'source observations pair with payload inventory')
        for a,c in zip(s,cuts[case]):
            require(a['ps']==c['ps'] and a['invalid']==c['loss']==c['invalid'] and a['rst']==c['bound'],'shared source observation identity')
        for a,b in zip(m,m[1:]):require(int(b['ps'])-int(a['ps'])==10000,'continuous management clock')
        gaps=[(int(a['ps']),int(b['ps'])) for a,b in zip(s,s[1:]) if int(b['ps'])-int(a['ps'])!=5000]
        if case==15:
            require(len(gaps)==1 and gaps[0][0]<events['stop_source']<events['resume_source']<=gaps[0][1],'only source gap is the injected clock stop')
        else:require(not gaps,'source clock remains running')
        require(all(x['req']=='0' or x['hrst']==x['srst']=='0' for x in m),'coordinated facing reset assertion')
        losses=[x for x in m if x['loss']=='1']
        if case==0:
            require(not losses and all(x['ack']==x['invalid']=='0' for x in s),'normal session remains valid')
            continue
        require(losses,'loss detected')
        first_loss=int(losses[0]['ps']);cause=expected[case]
        injection=min(t for name,t in events.items() if name.startswith(('inject_','stop_')))
        detection=first_loss-injection
        require(0<=detection<=(800000 if case in {9,10,11,13,14,15} else 80000),'bounded management loss detection')
        require(all(x['loss']=='1' and x['ha']==x['sa']=='0' for x in m if int(x['ps'])>=first_loss),'sticky loss and closed admission')
        require(all(x['causes']==str(cause) and x['tick']==losses[0]['tick'] for x in losses),'cause and first-event timestamp retained')
        first_invalid=int(next(x['ps'] for x in s if x['invalid']=='1'))
        first_ack=int(next(x['ps'] for x in s if x['ack']=='1'))
        require(first_ack>first_invalid,'owner ACK follows source invalidation')
        start_invalidation=events['resume_source'] if case==15 else first_loss
        require(0<=first_invalid-start_invalidation<=50000,'bounded source invalidation with available clock')
        if case==6:require(first_ack-first_invalid>=200000,'delayed owner ACK exercised with queued responses')
        spans=[];start=None
        for a,b in zip(m,m[1:]):
            t=int(b['ps'])
            if a['req']=='0' and b['req']=='1':
                require(t>first_loss and t-first_ack>=20000,'reset follows returned source ownership ACK')
                start=t
            if a['req']=='1' and b['req']=='0':
                require(start is not None,'complete reset interval')
                ns=sum(start<int(x['ps'])<t and x['bound']=='0' for x in cuts[case])
                nh=sum(start<int(x['ps'])<t and x['rst']=='0' for x in hosts[case])
                require(ns>=64 and nh>=64,'shared reset covers both domain clock requirements')
                spans.append(dict(start_ps=start,end_ps=t,source_edges=ns,host_edges=nh));start=None
        require(start is None and len(spans)==(2 if case==16 else 1),'all expected reset intervals')
        if case in {9,10,11,13,14}:
            require(spans[0]['start_ps']<events['resume_host']<spans[0]['end_ps'],'reset held through host clock restoration')
        if case==15:
            require(all(x['req']=='0' and x['hrst']==x['srst']=='1' for x in m if first_loss<=int(x['ps'])<events['resume_source']),'stopped source cannot authorize reset')
        require(m[-1]['state']=='8' and m[-1]['hrst']==m[-1]['srst']=='1','restored interfaces remain invalid')
        if case==14:
            when=events['host_received_before_source_ack']
            receipts=sum(int(x['ps'])<=when and num(x['hr'])&32!=0 and num(x['valid'])&32!=0 for x in hosts[case])
            outgoing=sum(int(x['ps'])<=when and num(x['mr'])&32!=0 and num(x['mv'])&32!=0 for x in cuts[case])
            require(receipts==1 and outgoing==0,'actual host receipt before producer ACK window')
            require(result['cases'][case]['delivered'][5]==1 and result['cases'][case]['abandoned'][5]==0,'return ACK resolves during guard detection before reset')
        checked.append(dict(case=case,cause=cause,detection_ps=detection,
                            source_invalidation_ps=first_invalid-start_invalidation,
                            ownership_ack_delay_ps=first_ack-first_invalid,reset_spans=spans))
    result['guard']=checked
    result['status']='PASS_AUTOMATIC_GUARD_WITH_REAL_CDC_17_CASES'
    return result
