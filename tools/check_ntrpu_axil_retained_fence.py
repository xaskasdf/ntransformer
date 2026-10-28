"""Conservation and permanent producer-fence evidence, independent of DUT state."""
import collections
import re


def review(text,period):
    def need(ok,message):
        if not ok:raise ValueError(message)
    def fields(line):return dict(re.findall(r'(\w+)=(\S+)',line))
    need(period in [5000,37000], 'clock profile')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M), 'simulator failure')
    need(text.splitlines().count(f'FENCE_SUITE_PASS cases=3 device_period_ps={period} native_endpoint=0 complete_dma_drain=0')==1,'scope')
    bus=collections.defaultdict(list);events=collections.defaultdict(list);passes={};resets=collections.defaultdict(list)
    for line in text.splitlines():
        if line.startswith('FENCE_BUS '):
            r=fields(line);side=r.pop('side');ch=r.pop('channel');case=int(r.pop('case'))
            need(case in range(3) and side in ['U','G','H','D'] and ch in ['AW','W','B','AR','R'],'bus identity')
            r={k:int(v,10 if k in ['ps','close'] else 16) for k,v in r.items()}
            need(r['ps']%(period if side=='D' else 3500)==(1100+period//2 if side=='D' else 700),'clock grid')
            bus[case,side,ch].append(r)
        if line.startswith('FENCE_EVENT '):
            r=fields(line);name=r.pop('name');r={k:int(v) for k,v in r.items()};r['name']=name;events[r['case']].append(r)
        if line.startswith('FENCE_CASE_PASS '):
            r={k:int(v) for k,v in fields(line).items()};need(r['case'] not in passes,'unique case pass');passes[r['case']]=r
        if line.startswith('AUTO_RESET '):
            r=fields(line);r={k:v if k=='event' else int(v,16 if k=='tag' else 10) for k,v in r.items()};resets[r['case']].append(r)
    need(set(passes)==set(range(3)),'three completed scenarios')
    results=[]
    def payload(rows,keys):return [tuple(r[k] for k in keys) for r in rows]
    names=['PLATEAU','PREFIX_PENDING','CERTIFIED_WITH_LOCAL_ERROR_PENDING','LOCAL_RESET_COMPLETE','FRESH_CLOCK_STOPPED','FRESH_CERTIFIED','TERMINAL']
    keys={'AW':['addr'],'W':['data','strb'],'B':['resp'],'AR':['addr'],'R':['data','resp']}
    for case in range(3):
        need(text.splitlines().count(f'FENCE_PROFILE case={case} device_period_ps={period} host_period_ps=7000 max_pending=32 observer_width=8')==1,'profile')
        ev=events[case];need([r['name'] for r in ev]==names,'ordered checkpoints')
        need(all(a['ps']<b['ps'] for a,b in zip(ev,ev[1:])),'checkpoint time order')
        plateau,pending,cert,after_reset,stopped,fresh,terminal=ev
        need(plateau['state']==0 and plateau['ack']==plateau['close']==0,'open saturation')
        if case in [0,2]:need(plateau['ga']==32,'AW cap')
        if case==0:need(plateau['gw']==plateau['gq']==0,'AW-only saturation')
        if case==1:need(plateau['ga']==plateau['gq']==0 and 0<plateau['gw']<32,'W-only downstream limit')
        if case==2:need(plateau['gw']==plateau['gq']==32,'simultaneous caps')
        need(pending['state']==1 and pending['ack']==pending['u_idle']==0 and pending['close']==1,'prefix cannot certify early')
        for e in [cert,after_reset,fresh]:
            need(e['state']==2 and e['fenced']==e['ack']==e['close']==e['shut']==1,'permanent section certificate')
            need(e['ga']==e['gw']==e['gq']==e['al']==e['wl']==e['rl']==0,'all downstream producer obligations zero')
            need(e['local_idle']==e['u_idle']==0 and e['bvalid']==1,'local rejection outside certified reset section')
        need(stopped['ack']==0 and stopped['fenced']==stopped['close']==1,'fresh generation masks old proof')
        need(terminal['ack']==0 and terminal['close']==terminal['fenced']==terminal['local_idle']==terminal['u_idle']==1,'no reopen on request withdrawal')
        u={ch:bus[case,'U',ch] for ch in keys};g={ch:bus[case,'G',ch] for ch in keys}
        wa=[0x7000+case*4]+[0x1000+case*1024+i*4 for i in range(128)]
        wd=[(0xa1b2c3d4+case,5)]+[(0xac000000 ^ (case<<16) ^ ((i*0x01030507)&0xffffffff),i%15+1) for i in range(128)]
        ra=[0x7000+case*4]+([i*4 for i in range(128)] if case==2 else [])
        need(payload(u['AW'],['addr'])==[(a,) for a in wa] and payload(u['W'],['data','strb'])==wd,'all upstream write offers')
        need(payload(u['AR'],['addr'])==[(a,) for a in ra],'all upstream read offers')
        ngw=len(g['AW']);ngr=len(g['AR']);need(1<ngw<129,'finite write prefix')
        need(ngr==(34 if case==2 else 1),'finite read prefix')
        need(payload(g['AW'],['addr'])==[(a,) for a in wa[:ngw]],'G write prefix addresses')
        need(payload(g['W'],['data','strb'])==wd[:ngw],'G paired write prefix')
        need(payload(g['AR'],['addr'])==[(a,) for a in ra[:ngr]],'G read prefix addresses')
        expected_b=[2 if r['close'] else (r['addr']>>2)&3 for r in g['AW']]
        need(payload(g['B'],['resp'])==[(v,) for v in expected_b],'manager response classification')
        need(payload(u['B'],['resp'])==[(v,) for v in expected_b+[2]*(129-ngw)],'manager/local write responses')
        original=0x13579bdf ^ (((0x1c00+case)*0x01020408)&0xffffffff)
        rd=[(original&0xff00ff00)|((0xa1b2c3d4+case)&0x00ff00ff)]+[0x13579bdf ^ ((i*0x01020408)&0xffffffff) for i in range(len(ra)-1)]
        need(len(g['R'])==ngr and len(u['R'])==len(ra),'all read responses')
        for i,r in enumerate(u['R']):
            local=i>=ngr;closed=local or g['AR'][i]['close']
            need(r['resp']==(2 if closed else (ra[i]>>2)&3),'U read response')
            if not closed:need(r['data']==rd[i],'forwarded read data including error responses')
            if local:need(r['data']==0,'local SLVERR read datum')
        need(payload(g['R'],['data','resp'])==payload(u['R'][:ngr],['data','resp']),'G read responses conserved')
        fw=[i for i,r in enumerate(g['AW']) if r['close']==0];fr=[i for i,r in enumerate(g['AR']) if r['close']==0]
        expected={'AW':[(wa[i],) for i in fw],'W':[wd[i] for i in fw],'B':[((wa[i]>>2)&3,) for i in fw],
                  'AR':[(ra[i],) for i in fr],'R':[(rd[i],(ra[i]>>2)&3) for i in fr]}
        for side in ['H','D']:
            for ch in keys:need(payload(bus[case,side,ch],keys[ch])==expected[ch],'downstream conservation '+side+ch)
        need(len(fw)==(1 if case==1 else 33) and len(fr)==(33 if case==2 else 1),'shutdown prefix classification')
        for side in ['G','H','D']:
            for ch in keys:need(all(r['ps']<cert['ps'] for r in bus[case,side,ch]),'no replay after isolation/reset '+side+ch)
        need(any(r['ps']>after_reset['ps'] for r in u['B']),'late rejection completes after reset')
        groups=collections.defaultdict(lambda:[0,0,0]);inv=[0,0,0];peak=[0,0,0]
        for ch,delta in [('AW',(1,0,0)),('W',(0,1,0)),('B',(-1,-1,0)),('AR',(0,0,1)),('R',(0,0,-1))]:
            for r in g[ch]:groups[r['ps']]=[a+b for a,b in zip(groups[r['ps']],delta)]
        for ps,delta in sorted(groups.items()):
            inv=[a+b for a,b in zip(inv,delta)];need(all(0<=a<=32 for a in inv),'hard credits/response causality')
            peak=[max(a,b) for a,b in zip(peak,inv)]
        need(inv==[0,0,0],'G independent final inventory')
        p=passes[case]
        need([p['max_aw'],p['max_w'],p['max_ar']]==peak,'independent maxima')
        need([p['uaw'],p['uar'],p['gaw'],p['gar'],p['daw'],p['dar']]==[129,len(ra),ngw,ngr,len(fw),len(fr)],'reported counters')
        rr=resets[case];need([r['event'] for r in rr]==['START','HELD','READY']*2,'two reset sequences')
        for j in [0,3]:
            start,held,ready=rr[j:j+3]
            need(start['ps']<held['ps']<ready['ps'] and start['tag']==held['tag']==ready['tag']-1,'reset identity/order')
            need(held['he']>=64 and held['de']>=64,'actual local reset hold edges')
        need(cert['ps']<rr[3]['ps']<rr[5]['ps']<after_reset['ps'],'certificate before operational reset')
        results.append(dict(case=case,upstream_writes=129,upstream_reads=len(ra),manager_writes=ngw,manager_reads=ngr,downstream_writes=len(fw),downstream_reads=len(fr),local_rejected_writes=129-ngw,local_rejected_reads=len(ra)-ngr,peak_pending=peak,events=ev,reset_phases=rr))
    need('$finish called at time' in text,'completed simulation')
    return dict(status='PASS_RETAINED_AXIL_PRODUCER_FENCE',device_period_ps=period,cases=results,scope='Retained admission producer, actual AMD manager/CDC, fresh management proof and local reset; upstream endpoint/internal DMA queues remain outside certificate.')
