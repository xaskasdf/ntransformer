"""Compare both real CDC interfaces and the distinct coordinated-reset gate."""
import re


def review(text,period):
    def need(ok,message):
        if not ok:raise ValueError(message)
    need(period in [5000,37000],'known clock profile')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'simulator failure')
    names=['aw_before_w_blocked_b','w_before_aw_masked_write','read_pending_blocked_r',
           'tail_and_coordinated_reset_clock_stop','post_reset_traffic_all_read_responses','fence_and_fresh_pair_identity']
    cases=re.findall(r'^PAIR_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$',text,re.M)
    need([(int(n),name) for n,name,ps in cases]==list(enumerate(names)),'six ordered cases')
    need(text.splitlines().count(f'PAIR_SUITE_PASS cases=6 writes=4 reads=4 device_period_ps={period} native_pcie=0 complete_endpoint_drain=0')==1,'scoped suite')
    need(text.splitlines().count(f'PAIR_PROFILE device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000')==1,'profile')
    buses={(side,ch):[] for side in ['H','D'] for ch in ['AW','W','B','AR','R']};events=[]
    for line in text.splitlines():
        if line.startswith('PAIR_BUS '):
            r=dict(re.findall(r'(\w+)=(\S+)',line));side=r.pop('side');ch=r.pop('channel')
            r={k:int(v,10 if k=='ps' else 16) for k,v in r.items()};buses[side,ch].append(r)
        if line.startswith('PAIR_EVENT '):
            r=dict(re.findall(r'(\w+)=(\S+)',line))
            events.append({k:v if k=='name' else int(v,16 if k=='native' else 10) for k,v in r.items()})
    fixtures={'AW':[dict(addr=v) for v in [0,0,4,8]],
        'W':[dict(data=d,strb=s) for d,s in [(0xdeadbeef,15),(0x11223344,5),(0xcafebabe,15),(0x76543210,15)]],
        'B':[dict(resp=0)]*4,'AR':[dict(addr=v) for v in [0,4,8,0]],
        'R':[dict(data=d,resp=r) for d,r in [(0xde22be44,3),(0xcafebabe,0),(0x76543210,1),(0xde22be44,2)]]}
    delays={}
    for ch,expected in fixtures.items():
        for side in ['H','D']:
            need([{k:v for k,v in r.items() if k!='ps'} for r in buses[side,ch]]==expected,'payload/count '+side+ch)
        samples=[]
        for h,d in zip(buses['H',ch],buses['D',ch]):
            need(h['ps']>d['ps'] if ch in ['AW','W','AR'] else d['ps']>h['ps'],'destination before source handshake '+ch)
            samples.append(abs(h['ps']-d['ps']))
        delays[ch]=dict(min_ps=min(samples),max_ps=max(samples),samples=4,meaning='receiver transfer to originating-interface acknowledgement, including clock stops')
    def event(case,name):
        rows=[r for r in events if r['case']==case and r['name']==name]
        need(len(rows)==1,'unique '+str(case)+' '+name);return rows[0]
    for case,name in [(0,'AW_ONLY'),(0,'B_BLOCKED'),(1,'W_ONLY'),(2,'AR_PENDING'),(2,'R_BLOCKED'),(5,'NO_FENCE'),(5,'NEW_TAG')]:
        need(event(case,name)['ack']==0,'no premature external certificate '+name)
    for case,name,values in [(0,'AW_ONLY',[1,0,0,1,0,0]),(1,'W_ONLY',[0,1,0,0,1,0]),(2,'AR_PENDING',[0,0,1,0,0,1])]:
        need([event(case,name)[k] for k in ['ha','hw','hq','da','dw','dq']]==values,'inventory '+name)
    drains=[r for r in events if r['name']=='DRAIN'];need(len(drains)==7,'seven fresh external certificates')
    for r in drains:
        need(r['ack']==r['hcert']==r['dcert']==1 and all(r[k]==0 for k in ['ha','hw','hq','da','dw','dq']),'external drain inventory')
    gap=event(3,'EXTERNAL_DRAIN_WITH_CLOCK_STOP')
    need(gap['ack']==1 and gap['hn']==gap['dn']==1,'drain is separate from reset')
    if period==37000:need(gap['native']!=0,'observed internal tail in slow profile')
    held=event(3,'HOLD_CLOCK_STOPPED')
    need(held['hn']==held['dn']==held['hd']==held['held_h']==0 and held['dd']==1 and held['held_d']>=64,'missing host edges prevent release')
    complete=event(3,'RESET_HELD_COMPLETE');released=event(3,'RESET_RELEASED')
    need(complete['hn']==complete['dn']==complete['native']==0 and complete['hd']==complete['dd']==1,'native state cleared before release')
    need(complete['held_h']>=64 and complete['held_d']>=64,'64 real cycles per domain')
    need(released['hn']==released['dn']==1 and released['native']==0 and released['ps']>complete['ps'],'synchronized release')
    for side in ['H','D']:
        need(buses[side,'AW'][3]['ps']>released['ps'] and buses[side,'R'][1]['ps']>released['ps'],'verified traffic after reset '+side)
    need('$finish called at time' in text,'simulation completion')
    return dict(status='PASS_AXIL_PAIR_EXTERNAL_DRAIN_AND_COORDINATED_RESET',device_period_ps=period,cases=names,
        writes=4,reads=4,handshake_ack_delays=delays,native_state_at_external_certificate=gap['native'],
        reset_held_edges=dict(host=complete['held_h'],device=complete['held_d']),
        scope='Actual AMD AXI4-Lite clock converter, both retained ledgers and XPM certificates. Coordinated reset/hold/release driven by the bench using real reset primitives. Host isolation is a controlled contract; no native PCIe, complete endpoint drain or automatic reset coordinator approval.')
