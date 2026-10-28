"""Independent payload, conservation, saturation and rejection trace review."""
import collections
import re


def review(text, period):
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    def row(line):
        return dict(re.findall(r'(\w+)=(\S+)', line))
    need(period in (5000, 37000), 'clock profile')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )', text, re.M), 'simulation failure')
    need(text.splitlines().count(f'SAT_PROFILE device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000 observer_width=8 offers=128') == 1, 'profile/observer')
    need(text.splitlines().count(f'SAT_SUITE_PASS cases=5 writes=517 reads=261 device_period_ps={period} native_endpoint=0 complete_dma_drain=0') == 1, 'bounded scope')
    buses = collections.defaultdict(list)
    plateaus = []
    reopens = []
    passes = []
    events = []
    for line in text.splitlines():
        if line.startswith('SAT_BUS '):
            r = row(line);side = r.pop('side');ch = r.pop('channel')
            r = {k:int(v, 10 if k in ('case','close','ps') else 16) for k,v in r.items()}
            need(side in ('U','H','D') and ch in ('AW','W','B','AR','R'), 'bus identity')
            need(r['ps'] % (period if side == 'D' else 7000) == (1100+period//2 if side == 'D' else 4200), 'rising edge '+side+ch)
            buses[r['case'],side,ch].append(r)
        if line.startswith('SAT_PLATEAU '):plateaus.append({k:int(v) for k,v in row(line).items()})
        if line.startswith('SAT_REOPEN '):reopens.append({k:int(v) for k,v in row(line).items()})
        if line.startswith('SAT_CASE_PASS '):passes.append({k:int(v) for k,v in row(line).items()})
        if line.startswith('SHUT_EVENT '):
            events.append({k:v if k=='name' else int(v,16 if k=='native' else 10) for k,v in row(line).items()})
    need([r['case'] for r in plateaus] == list(range(5)), 'five saturation probes')
    need([r['case'] for r in reopens] == list(range(5)), 'five reopen probes')
    need([r['case'] for r in passes] == list(range(5)), 'five completed cases')
    need(len(re.findall(r'^AUTO_RESET event=START ',text,re.M)) == 1, 'configuration reset only')
    counts = [];measured = [];peaks = [0,0,0]
    def payload(rows, keys):
        return [tuple(r[k] for k in keys) for r in rows]
    for case in range(5):
        u = {ch:buses[case,'U',ch] for ch in ('AW','W','B','AR','R')}
        nw = 0 if case==3 else 128;nr = 128 if case>=3 else 0
        wa = [0x1000+case*1024+n*4 for n in range(nw)] + [0x7000+case*4]
        wd = [(0xac000000 ^ (case<<16) ^ ((n*0x01030507)&0xffffffff), n%15+1) for n in range(nw)] + [(0xa1b2c3d4+case,5)]
        ra = [n*4 for n in range(nr)] + [0x7000+case*4]
        need(payload(u['AW'],['addr']) == [(v,) for v in wa], 'offered write addresses')
        need(payload(u['W'],['data','strb']) == wd, 'offered write payloads/masks')
        need(payload(u['AR'],['addr']) == [(v,) for v in ra], 'offered read addresses')
        fw = [i for i,r in enumerate(u['AW']) if not r['close']]
        fr = [i for i,r in enumerate(u['AR']) if not r['close']]
        need(all(r['close'] in (0,1) for r in u['AW']+u['AR']), 'binary close classification')
        need(fw[-1] == nw and fr[-1] == nr, 'reopened probe admitted')
        if nw:need(len(fw)<len(wa), 'late writes rejected')
        if nr:need(len(fr)<len(ra), 'late reads rejected')
        if case==1:need(fw==[nw], 'W before AW is never forwarded on close')
        initial = (0x13579bdf ^ ((ra[-1]//4*0x01020408)&0xffffffff))
        probe_data = (initial & 0xff00ff00) | ((0xa1b2c3d4+case)&0x00ff00ff)
        rd = [(0x13579bdf ^ ((a//4*0x01020408)&0xffffffff)) for a in ra[:-1]] + [probe_data]
        expected = {'AW':[(wa[i],) for i in fw], 'W':[wd[i] for i in fw],
                    'B':[((wa[i]>>2)&3,) for i in fw], 'AR':[(ra[i],) for i in fr],
                    'R':[(rd[i],(ra[i]>>2)&3) for i in fr]}
        keys = {'AW':['addr'],'W':['data','strb'],'B':['resp'],'AR':['addr'],'R':['data','resp']}
        for side in ('H','D'):
            for ch in keys:need(payload(buses[case,side,ch],keys[ch])==expected[ch], 'forwarded conservation '+side+ch)
        need(payload(u['B'],['resp']) == [(2 if r['close'] else (r['addr']>>2)&3,) for r in u['AW']], 'all write responses including forwarded errors')
        need(len(u['R'])==len(ra), 'all read completions')
        for i,r in enumerate(u['R']):
            closed=u['AR'][i]['close']
            need(r['resp'] == (2 if closed else (ra[i]>>2)&3), 'read response classification')
            if not closed:need(r['data']==rd[i], 'forwarded data even on error response')
        p=plateaus[case];reopen=reopens[case]
        need(p['quiet']==256 and p['ps']<reopen['ps']<passes[case]['ps'], 'probe ordering')
        accepted = {ch:sum(r['ps']<=p['ps'] for r in u[ch]) for ch in u}
        need(accepted['B']==accepted['R']==0, 'upstream completions blocked')
        need([p['aw'],p['w'],p['ar']] == [accepted['AW'],accepted['W'],accepted['AR']], 'measured acceptance counts')
        need([p['ua'],p['uw'],p['uq']] == [p['aw'],p['w'],p['ar']], 'wide observer agrees with independent trace')
        for ch,prefix,active in [('AW','aw',case not in (1,3)),('W','w',case not in (0,3)),('AR','ar',case>=3)]:
            need(p[prefix+'v']==int(active) and p[prefix+'r']==0 if active else p[prefix+'v']==0, 'held saturated '+ch)
            need(0<p[prefix]<128 if active else p[prefix]==0, 'saturation bounded '+ch)
            need(not any(p['ps']-256*7000 < r['ps'] <= p['ps'] for r in u[ch]), 'plateau quiet '+ch)
        need(reopen['idle']==1, 'reopen idle')
        for side in ('U','H','D'):
            for ch in ('AW','W','AR'):
                need(buses[case,side,ch][-1]['ps']>reopen['ps'], 'fresh probe after reopen')
        closed_events=[r for r in events if r['case']==case and r['name']=='SAT_CLOSED_STILL_BLOCKED']
        drains=[r for r in events if r['case']==case and r['name']=='SAT_DRAIN']
        need(len(closed_events)==len(drains)==1, 'closure and drain evidence')
        drain=drains[0]
        need(drain['up_idle']==drain['ack']==drain['shut']==drain['manager_reset_n']==drain['hn']==drain['dn']==1, 'drained without reset')
        need(all(drain[k]==0 for k in ('ua','uw','uq','ha','hw','hq','da','dw','dq')), 'zero inventory at drain')
        need(p['ps']<closed_events[0]['ps']<drain['ps']<reopen['ps'], 'closure ordering')
        groups=collections.defaultdict(lambda:[0,0,0])
        for ch,delta in [('AW',(1,0,0)),('W',(0,1,0)),('B',(-1,-1,0)),('AR',(0,0,1)),('R',(0,0,-1))]:
            for r in u[ch]:groups[r['ps']]=[v+d for v,d in zip(groups[r['ps']],delta)]
        inv=[0,0,0]
        for ps,delta in sorted(groups.items()):
            inv=[v+d for v,d in zip(inv,delta)]
            need(all(0<=v<256 for v in inv), 'no response before acceptance or overflow')
            peaks=[max(v,w) for v,w in zip(peaks,inv)]
        need(inv==[0,0,0], 'independent terminal inventory')
        counts.append(dict(case=case,upstream_writes=nw+1,upstream_reads=nr+1,downstream_writes=len(fw),downstream_reads=len(fr),rejected_writes=nw+1-len(fw),rejected_reads=nr+1-len(fr)))
        measured.append(p)
    maxima=re.findall(r'^SAT_MAX aw=(\d+) w=(\d+) ar=(\d+)$',text,re.M)
    need(len(maxima)==1 and list(map(int,maxima[0]))==peaks, 'peak observer independently reconstructed')
    need(max(peaks)>15, 'wider observer actually exercised')
    stalls=re.findall(r'^SAT_STALLS uaw=(\d+) uw=(\d+) uar=(\d+) hb=(\d+) hr=(\d+)$',text,re.M)
    need(len(stalls)==1 and all(int(v)>0 for v in stalls[0]), 'all request/response stalls')
    need('$finish called at time' in text, 'simulation completed')
    need(sum(x['upstream_writes'] for x in counts)==517 and sum(x['upstream_reads'] for x in counts)==261, 'suite totals')
    return dict(status='PASS_AMD_SHUTDOWN_SATURATION',device_period_ps=period,counts=counts,plateaus=measured,peak_outstanding=dict(zip(['AW','W','AR'],peaks)),stall_cycles=list(map(int,stalls[0])),reset_scope='configuration only',scope='AXI-Lite actual manager/CDC with behavioral producers/subordinate; not full XDMA drain or physical timing.')
