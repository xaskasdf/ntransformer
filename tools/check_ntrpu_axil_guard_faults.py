"""Independent conservation and source-clock audit for exclusive guard faults."""
import collections
import re


def review(text,period,stop_source):
    def need(ok,message):
        if not ok:
            raise ValueError(message)
    def fields(line):
        return dict(re.findall(r'(\w+)=(\S+)',line))
    def numeric(row):
        return {k:int(v,16 if k in ['rdata','data','addr','strb','resp'] else 10) for k,v in row.items()}
    need(period in [5000,37000] and stop_source in [0,1],'clock profile')
    need(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|CRITICAL WARNING:|FAIL )',text,re.M),'simulator failure')
    marker=f'GUARD_FAULT_SUITE_PASS cases=6 device_period_ps={period} stop_source={stop_source} native_endpoint=0 complete_dma_drain=0'
    need(text.splitlines().count(marker)==1 and '$finish called at time' in text,'scoped complete suite')
    events=collections.defaultdict(list);bus=collections.defaultdict(list);raw=collections.defaultdict(list)
    resets=collections.defaultdict(list);passes={}
    for line in text.splitlines():
        if line.startswith('GUARD_EVENT '):
            row=fields(line);name=row.pop('name');row=numeric(row);row['name']=name;events[row['case']].append(row)
        elif line.startswith('GUARD_CASE_PASS '):
            row=numeric(fields(line));need(row['case'] not in passes,'unique pass');passes[row['case']]=row
        elif line.startswith('FENCE_BUS '):
            row=fields(line);case=int(row.pop('case'));side=row.pop('side');channel=row.pop('channel');row=numeric(row)
            need(case in range(6) and side in ['U','G','H','D'] and channel in ['AW','W','B','AR','R'],'bus identity')
            step,phase=(period//2,1100) if side=='D' else (3500,700)
            need(row['ps']%step==phase,'edge clock grid')
            bus[case,side,channel].append(row)
        elif line.startswith('GUARD_RAW '):
            row=fields(line);channel=row.pop('channel');row=numeric(row);row['channel']=channel;raw[row['case']].append(row)
        elif line.startswith('AUTO_RESET '):
            row=fields(line);resets[int(row['case'])].append(row)
    need(set(events)==set(raw)==set(passes)==set(range(6)),'six complete cases')
    results=[]
    for case in range(6):
        native=case in [2,3];read=case in [1,2,5];post=case>=4
        profile=(f'GUARD_PROFILE case={case} device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000 '
                 f'stop_source={stop_source} native_injection={int(native)} read_orphan={int(read)} post_certificate={int(post)}')
        need(text.splitlines().count(profile)==1,'actual mode identity')
        rows=events[case]
        need([r['name'] for r in rows]==['BASELINE','ARMED','SOURCE_FAULT','PROPAGATED','FRESH_BLOCKED','RESUMED','SOURCES_CLEARED','TERMINAL'],'ordered phases')
        need(all(a['ps']<b['ps'] for a,b in zip(rows,rows[1:])),'phase chronology')
        baseline,armed,source,propagated,fresh,resumed,cleared,terminal=rows
        fault=12 if native else 8;expected_sources=2 if read else 1
        need(baseline['ack']==armed['ack']==baseline['close']==int(post),'pre/post-certificate baseline')
        need(baseline['req']==int(post) and baseline['guard_sources']==baseline['mgmt']==baseline['guard_ps']==0,'no initial fault')
        first,report,revoked=terminal['guard_ps'],terminal['report_ps'],terminal['revoked_ps']
        need(armed['ps']<first and source['ps']==first+2,'source register observed without extra edge')
        need(0<report-first<=30000 and report%10000==5000,'management latency and phase')
        need(revoked==(report if post else 0),'guard-only cached certificate revocation')
        need(all(r['guard_ps']==first for r in rows[2:]),'stable source timestamp')
        need(all(r['report_ps']==report and r['revoked_ps']==revoked for r in rows[3:]),'stable management measurements')
        need(all(r['ack']==r['upstream']==r['pair']==0 and r['mgmt']==fault and r['close']==1 for r in rows[3:]),'retained exclusive guard authorization block')
        need(all(r['guard_sources']==expected_sources for r in rows[2:6]+rows[7:]),'actual retained source identity')
        need(all(r['gate_fault']==r['h_fault']==0 for r in rows),'no substitute admission/H fault')
        need(all(r['d_fault']==int(native) for r in rows[2:6]+rows[7:]),'D source only in native-injection cases')
        need(cleared['guard_sources']==cleared['d_fault']==0,'actual source reports removed')
        need(all(r['hn']==r['dn']==r['device_run']==1 and r['ua']==r['uw']==r['uq']==r['busy']==0 for r in rows),'no reset, leftover upstream obligation or new mailbox')
        need(all(r['host_run']==1-stop_source for r in rows[2:5]),'host clock mode observed')
        need(all(r['host_run']==1 for r in rows[5:]),'host resumes')
        if stop_source:
            need(all(r['stopped_ps']==first+1 and r['host_edges']==r['edges_at_fault']==source['host_edges'] for r in rows[2:5]),'zero host edges during report and fresh request')
            need(resumed['host_edges']>source['host_edges'],'real clock resumption')
        else:
            need(all(r['stopped_ps']==0 for r in rows),'no hidden clock stop')
            need(fresh['host_edges']>source['host_edges'],'clock actually runs')
        need(fresh['req']==1 and fresh['generation']==armed['generation']+1 and fresh['tag']==armed['tag']+1,'fresh identity exercised')
        for row in rows[2:]:
            need(row['hbv']==row['hrv']==row['br']==row['rr']==0,'no raw acceptance or protected response')
            if read:
                need(row['rv']==1 and row['rresp']==3 and row['rdata']==0xbad3c0de and row['bv']==0,'retained orphan R payload')
            else:
                need(row['bv']==1 and row['bresp']==2 and row['rv']==0,'retained orphan B payload')
        address=0x7000+case*4;datum=0xa1b2c3d4+case
        old=0x13579bdf ^ (((0x1c00+case)*0x01020408)&0xffffffff)
        readback=(old&0xff00ff00)|(datum&0x00ff00ff)
        expected={'AW':[(address,)],'W':[(datum,5)],'B':[(case%4,)],'AR':[(address,)],'R':[(readback,case%4)]}
        keys={'AW':['addr'],'W':['data','strb'],'B':['resp'],'AR':['addr'],'R':['data','resp']}
        for side in ['U','G','H','D']:
            for channel,k in keys.items():
                want=list(expected[channel])
                if native and side=='D' and channel==('R' if read else 'B'):
                    want.append((0xbad3c0de,3) if read else (2,))
                got=[tuple(r[x] for x in k) for r in bus[case,side,channel]]
                need(got==want,f'complete request/response inventory {case}/{side}/{channel}')
        rr=raw[case]
        need(len(rr)==3 and [x['channel'] for x in rr]==['B','R','R' if read else 'B'],'exact raw response offers')
        need(rr[0]['aw']==rr[0]['w']==rr[1]['ar']==1 and rr[0]['protected_valid']==rr[1]['protected_valid']==1,'healthy warmup has real prior credit')
        orphan=rr[-1]
        need(orphan['ps']==first and orphan['ready']==orphan['protected_valid']==0,'orphan blocked on first sampled offer')
        if read:
            need(orphan['ar']==0 and orphan['data']==0xbad3c0de and orphan['resp']==3,'raw orphan R without credit')
        else:
            need(orphan['aw']==orphan['w']==0 and orphan['resp']==2,'raw orphan B without both write halves')
        if native:
            d=bus[case,'D','R' if read else 'B'][-1]
            need(armed['ps']<d['ps']<first,'native injection precedes host quarantine')
        counts={key:len(bus[case,side,ch]) for key,side,ch in [
            ('uaw','U','AW'),('uar','U','AR'),('ubc','U','B'),('urc','U','R'),
            ('haw','H','AW'),('har','H','AR'),('hbc','H','B'),('hrc','H','R'),
            ('daw','D','AW'),('dar','D','AR'),('dbc','D','B'),('drc','D','R')]}
        need(passes[case]==dict(case=case,guard_ps=first,report_ps=report,revoked_ps=revoked,mgmt=fault,**counts),'pass counts reconstructed from every bus')
        reset=resets[case];need([r['event'] for r in reset]==['START','HELD','READY'],'configuration reset only')
        need(int(reset[1]['he'])>=64 and int(reset[1]['de'])>=64 and int(reset[-1]['ps'])<baseline['ps'],'startup reset completed')
        results.append(dict(case=case,source='R' if read else 'B',native_injection=native,post_certificate=post,
                            report_latency_ps=report-first,host_edges_during_fault=fresh['host_edges']-source['host_edges'],
                            events=rows,raw_offers=rr,counts=counts))
    return dict(status='PASS_SIX_GUARD_FAULT_CASES',period_ps=period,stop_source=stop_source,
                cases=results,native_endpoint=False,complete_dma_drain=False)
