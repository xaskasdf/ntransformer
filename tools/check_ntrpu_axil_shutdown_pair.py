"""Check the actual shutdown manager across upstream, CDC input and device."""
import re


def review(text, period):
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    need(period in [5000, 37000], 'clock profile')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )', text, re.M), 'simulator failure')
    names = ['aw_before_shutdown_late_w_preserved', 'w_before_shutdown_aw_after_rejected',
             'prior_read_and_held_response_preserved', 'reject_new_traffic_without_downstream_access',
             'retained_shutdown_through_automatic_local_reset', 'post_reset_payload_masks_and_responses']
    cases = re.findall(r'^SHUT_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$', text, re.M)
    need([(int(n), name) for n, name, _ in cases] == list(enumerate(names)), 'six ordered cases')
    need(text.splitlines().count(f'SHUT_SUITE_PASS cases=6 upstream_writes=6 upstream_reads=5 downstream_writes=4 downstream_reads=4 device_period_ps={period} native_endpoint=0 complete_dma_drain=0') == 1, 'scope')
    need(text.splitlines().count(f'SHUT_PROFILE device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000') == 1, 'profile')
    buses = {(side, ch): [] for side in ['U', 'H', 'D'] for ch in ['AW','W','B','AR','R']}
    events = []
    for line in text.splitlines():
        if line.startswith('SHUT_BUS '):
            r = dict(re.findall(r'(\w+)=(\S+)', line));side = r.pop('side');ch = r.pop('channel')
            buses[side,ch].append({k:int(v,10 if k=='ps' else 16) for k,v in r.items()})
        if line.startswith('SHUT_EVENT '):
            r = dict(re.findall(r'(\w+)=(\S+)', line))
            events.append({k:v if k=='name' else int(v,16 if k=='native' else 10) for k,v in r.items()})
    fixtures = {'AW': [dict(addr=v) for v in [0,4,8,0]],
                'W': [dict(data=d,strb=m) for d,m in [(0xdeadbeef,15),(0xcafebabe,15),(0x76543210,15),(0x11223344,5)]],
                'B': [dict(resp=0)]*4, 'AR':[dict(addr=v) for v in [0,4,8,0]],
                'R': [dict(data=d,resp=r) for d,r in [(0xdeadbeef,3),(0xcafebabe,0),(0x76543210,1),(0xde22be44,2)]]}
    for side in ['H','D']:
        for ch, expected in fixtures.items():
            need([{k:v for k,v in r.items() if k!='ps'} for r in buses[side,ch]] == expected, 'payload/count '+side+ch)
    upstream = {'AW':[dict(addr=v) for v in [0,0,8,4,8,0]],
                'W':[dict(data=d,strb=m) for d,m in [(0xdeadbeef,15),(0xbad0bad0,15),(0xbeefbeef,15),(0xcafebabe,15),(0x76543210,15),(0x11223344,5)]],
                'B':[dict(resp=v) for v in [0,2,2,0,0,0]],
                'AR':[dict(addr=v) for v in [0,12,4,8,0]],
                'R':[dict(data=0xdeadbeef,resp=3),dict(resp=2),dict(data=0xcafebabe,resp=0),dict(data=0x76543210,resp=1),dict(data=0xde22be44,resp=2)]}
    for ch, expected in upstream.items():
        actual=buses['U',ch]
        need(len(actual)==len(expected),'upstream count '+ch)
        for got,want in zip(actual,expected):
            need(all(got[k]==v for k,v in want.items()),'upstream payload '+ch)
    for (side,ch),rows in buses.items():
        for r in rows:
            need(r['ps']%(3500 if side in ['U','H'] else period//2)==(700 if side in ['U','H'] else 1100),'clock edge '+side+ch)
    def event(case,name):
        found=[r for r in events if r['case']==case and r['name']==name]
        need(len(found)==1,'unique '+name);return found[0]
    for case,name,fields in [
        (0,'AW_ONLY',dict(ua=1,uw=0,up_idle=0,ack=0)),
        (0,'DOWNSTREAM_BLOCKED',dict(up_idle=0,ack=0)),
        (0,'UPSTREAM_B_BLOCKED',dict(ua=1,uw=1,up_idle=0)),
        (1,'W_WITHOUT_AW',dict(uw=1,ua=0,up_idle=0)),
        (2,'AR_PENDING',dict(uq=1,up_idle=0,ack=0)),
        (2,'UPSTREAM_R_BLOCKED',dict(uq=1,up_idle=0)),
        (3,'REJECT_RESPONSE_PENDING',dict(ack=1,shut=1,up_idle=0,uq=1)),
        (3,'REJECTED_WHILE_FENCED',dict(shut=1,up_idle=1)),
        (4,'DRAIN_BEFORE_RESET',dict(ack=1,shut=1,up_idle=1,hn=1,dn=1)),
        (4,'HOLD_CLOCK_STOPPED',dict(hn=0,dn=0,hd=0,dd=1,held_h=0,manager_reset_n=1)),
        (4,'RESET_RELEASED',dict(hn=1,dn=1,shut=1,native=0,manager_reset_n=1)),
    ]:
        got=event(case,name);need(all(got[k]==v for k,v in fields.items()),name+' fields')
    phases=[]
    for line in text.splitlines():
        if line.startswith('AUTO_RESET '):
            r=dict(re.findall(r'(\w+)=(\S+)',line));phases.append({k:v if k=='event' else int(v,16 if k=='tag' else 10) for k,v in r.items()})
    need([r['event'] for r in phases]==['START','HELD','READY']*2,'two automatic resets')
    for i in [0,3]:
        start,held,ready=phases[i:i+3]
        need(start['ps']<held['ps']<ready['ps'] and start['tag']==held['tag']==ready['tag']-1,'reset order and identity')
        need(held['he']>=64 and held['de']>=64,'real reset edge counts')
    released=event(4,'RESET_RELEASED')['ps']
    for side in ['U','H','D']:
        need(buses[side,'AR'][-1]['ps']>released and buses[side,'AW'][-1]['ps']>released,'post-reset traffic '+side)
    stalls=re.findall(r'^SHUT_STALLS aw=(\d+) w=(\d+) ar=(\d+) b=(\d+) r=(\d+)$',text,re.M)
    need(len(stalls)==1 and all(int(v)>0 for v in stalls[0]),'all five channels stalled')
    need('$finish called at time' in text,'completion')
    return dict(status='PASS_AMD_SHUTDOWN_AND_ACTUAL_CDC',device_period_ps=period,cases=names,
                upstream_writes=6,upstream_reads=5,downstream_writes=4,downstream_reads=4,
                rejected_writes=2,rejected_reads=1,events=events,reset_phases=phases,
                stalled_cycles=dict(zip(['AW','W','AR','B','R'],map(int,stalls[0]))),
                scope='AMD retained shutdown manager plus real CDC and local actuator. Upstream producer control and protected XDMA queues remain outside this certificate.')
