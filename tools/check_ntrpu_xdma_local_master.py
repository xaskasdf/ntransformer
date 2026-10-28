"""Black-box VCD scoreboard for local-master transaction and timeout obligations."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows

# write, address, write data, strobes, response data, response code, timed out
CASES = {
    1: (0, 0x40, 0, 0, 0x47, 0, 0),
    2: (1, 0x0c, 1, 15, 0, 0, 0),
    3: (1, 0x100c, 1, 1, 0, 2, 0),
    4: (1, 0x4080, 0x12345678, 10, 0, 3, 0),
    5: (0, 0x1048, 0, 0, 0x89abcdef, 2, 0),
    6: (0, 0x48, 0, 0, 0x21, 0, 1),
    7: (1, 0x0c, 1, 15, 0, 0, 1),
    8: (1, 0x100c, 1, 15, 0, 0, 1),
    9: (1, 0x0c, 1, 15, 0, 2, 1),
    10: (0, 0x1004, 0, 0, 0x00fffe7e, 0, 1),
}
FIELDS = {'AW': ['awaddr', 'awprot'], 'W': ['wdata', 'wstrb'],
          'B': ['bresp'], 'AR': ['araddr', 'arprot'], 'R': ['rdata', 'rresp']}


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    require(markers(sim, 'LOCAL_MASTER_PASS') == [dict(cases='10', native_xdma='0', reset_permission='0')], 'bounded unit scope')
    require('$finish called at time : 1564 ns' in sim, 'bounded simulator finish')
    prefix = '/tb_ntrpu_xdma_local_master/'
    events = {n: [] for n in ['REQ', 'AW', 'W', 'B', 'AR', 'R', 'RSP']}
    pending, sticky, reset_rises, last_edge = None, 0, [], None
    completed, timeouts = [], []
    stalls = {c: 0 for c in FIELDS}
    held = {}

    def event(name, t, phase, **values):
        events[name].append(dict(t=t, phase=phase, **values))

    for t, raw_before, raw_after in vcd_rows(vcd):
        before = {n[len(prefix):]: v for n, v in raw_before.items() if n.startswith(prefix)}
        after = {n[len(prefix):]: v for n, v in raw_after.items() if n.startswith(prefix)}
        if not after or all(v is None for v in after.values()):
            continue  # The parser yields an empty initial timestamp before $dumpvars.
        require(all(v is not None for v in after.values()), 'unknown public signal')
        if before.get('cold_rst_n') == 0 and after['cold_rst_n'] == 1:
            reset_rises.append(t)
        if before.get('cold_rst_n') == 1 and after['cold_rst_n'] == 0:
            require(pending is None, 'cold configuration cannot cancel outstanding operation')
        edge = before.get('clk') == 0 and after['clk'] == 1
        if edge:
            if last_edge is not None:
                require(t-last_edge == 4000, 'continuous 250MHz unit clock')
            last_edge = t
            if not before['cold_rst_n']:
                require(pending is None, 'configuration while busy')
                sticky, held = 0, {}
            else:
                phase = before['phase']
                for c, payload in held.items():
                    stem = 'm_axil_'+c.lower()
                    require(before[stem+'valid'] == 1 and
                            [before['m_axil_'+f] for f in FIELDS[c]] == payload, 'retained '+c)
                held = {}
                for c in FIELDS:
                    stem = 'm_axil_'+c.lower()
                    if before[stem+'valid'] and not before[stem+'ready']:
                        held[c] = [before['m_axil_'+f] for f in FIELDS[c]]
                        stalls[c] += 1
                if before['req_valid'] and before['req_ready']:
                    require(pending is None and not sticky and before['admit'], 'single outstanding admitted request')
                    require(phase == len(completed)+1 and phase in CASES, 'request order')
                    spec = CASES[phase]
                    identity = (0x1020304050607080 << 64) | phase
                    require([before[n] for n in ['req_write','req_address','req_data','req_strobe','req_identity']] ==
                            [*spec[:4], identity], 'request payload and identity')
                    pending = dict(phase=phase, start=t, identity=identity, spec=spec,
                                   aw=None, w=None, ar=None, bus_response=None, response=None,
                                   timeout=None, report_stalls=0)
                    event('REQ', t, phase, write=spec[0], addr=spec[1], data=spec[2], strb=spec[3], identity=identity)
                for c in FIELDS:
                    stem = 'm_axil_'+c.lower()
                    if not (before[stem+'valid'] and before[stem+'ready']):
                        continue
                    require(pending is not None and pending['bus_response'] is None, 'unexpected bus response/request '+c)
                    p, spec = pending, pending['spec']
                    require(phase == p['phase'], 'phase of accepted transaction')
                    if c in ('AW', 'W', 'AR'):
                        require(p[c.lower()] is None and spec[0] == int(c != 'AR'), 'exactly one address/data handshake')
                        if c == 'W':
                            require([before['m_axil_wdata'], before['m_axil_wstrb']] == [spec[2], spec[3]], 'captured write data/strobes')
                            event(c,t,phase,data=spec[2],strb=spec[3])
                        else:
                            require([before[stem+'addr'],before[stem+'prot']] == [spec[1],0], 'captured address/protection')
                            event(c,t,phase,addr=spec[1],prot=0)
                        p[c.lower()] = t
                    else:
                        require(spec[0] == int(c == 'B'), 'response type')
                        require((p['aw'] is not None and p['w'] is not None) if c == 'B' else p['ar'] is not None, 'response after requests')
                        require(before[stem+'resp'] == spec[5], 'real bus error code')
                        if c == 'R':
                            require(before['m_axil_rdata'] == spec[4], 'real bus read data')
                            event(c,t,phase,data=spec[4],resp=spec[5])
                        else:
                            event(c,t,phase,resp=spec[5])
                        p['bus_response'] = t
                        p['response'] = [spec[4],spec[5],p['identity'],int(sticky or t-p['start'] >= 128000)]
                if pending is not None:
                    p = pending
                    if (p['bus_response'] is None or p['bus_response'] == t) and t-p['start'] >= 128000:
                        if not sticky:
                            require(t-p['start'] == 128000, 'timeout on exact 32nd cycle')
                            p['timeout'] = t
                            channels = (['AW','W'] if p['spec'][0] else ['AR'])
                            waiting = [c for c in channels if p[c.lower()] is None]
                            timeouts.append(dict(phase=phase,t=t,pending_channels=waiting or
                                                 (['B'] if p['spec'][0] else ['R'])))
                        sticky = 1
                    if before['rsp_valid'] and not before['rsp_ready']:
                        p['report_stalls'] += 1
                    if before['rsp_valid'] and before['rsp_ready']:
                        require(p['bus_response'] is not None and t > p['bus_response'], 'report follows real response')
                        require([before[n] for n in ['rsp_data','rsp_code','rsp_identity','rsp_timeout']] == p['response'], 'report data identity error timeout')
                        require(before['rsp_timeout'] == p['spec'][6], 'scenario timeout expectation')
                        event('RSP',t,phase,data=p['response'][0],resp=p['response'][1],identity=p['identity'],timeout=p['response'][3])
                        completed.append(dict(phase=phase,request_ps=p['start'],aw_ps=p['aw'],w_ps=p['w'],ar_ps=p['ar'],
                                              response_ps=p['bus_response'],report_ps=t,timeout_ps=p['timeout'],
                                              report_stall_cycles=p['report_stalls'],request_to_bus_ps=p['bus_response']-p['start']))
                        pending = None
                else:
                    require(not before['rsp_valid'], 'no invented completion')
        # Continuous public contract: no dependence on internal FSM state.
        require(after['timeout_latched'] == sticky, 'sticky timeout and cold-only recovery')
        require(after['busy'] == int(pending is not None), 'busy covers request and retained report')
        require(after['req_ready'] == int(after['cold_rst_n'] and after['admit'] and pending is None and not sticky), 'admission predicate')
        response = pending is not None and pending['bus_response'] is not None
        require(after['rsp_valid'] == int(response), 'publish only actual bus responses')
        if response:
            require([after[n] for n in ['rsp_data','rsp_code','rsp_identity','rsp_timeout']] == pending['response'], 'stable retained report')
        for c in ('AW','W','AR'):
            wanted = pending is not None and pending['spec'][0] == int(c != 'AR') and pending[c.lower()] is None
            require(after['m_axil_'+c.lower()+'valid'] == int(wanted), 'request remains offered '+c)
        for c in ('B','R'):
            wanted = pending is not None and not response and pending['spec'][0] == int(c == 'B')
            if wanted:
                wanted = (pending['aw'] is not None and pending['w'] is not None) if c == 'B' else pending['ar'] is not None
            require(after['m_axil_'+c.lower()+'ready'] == int(wanted), 'accept real late response '+c)
    require(pending is None and len(completed) == 10, 'all obligations retired')
    require(len(reset_rises) == 6, 'six independent configuration lifetimes')
    require([r['phase'] for r in timeouts] == [6,7,8,9,10], 'five distinct timeout sites')
    by = {r['phase']:r for r in completed}
    require(by[2]['aw_ps'] < by[2]['w_ps'] and by[3]['w_ps'] < by[3]['aw_ps'] and by[4]['aw_ps'] == by[4]['w_ps'], 'AW/W permutations')
    require(by[5]['report_stall_cycles'] > 32 and by[5]['timeout_ps'] is None, 'report backpressure does not expire transaction')
    require(all(stalls[c] > 0 for c in ['AW','W','AR']), 'request backpressure exercised')
    for name, actual in events.items():
        parsed = []
        for row in markers(sim,'LOCAL_MASTER_'+name):
            parsed.append({k:int(v,10 if k in ('t','phase','write','timeout') else 16) for k,v in row.items()})
        require(parsed == actual, 'log/VCD correspondence '+name)
    cases = markers(sim,'LOCAL_MASTER_CASE')
    require(len(cases) == 10, 'ten case markers')
    for phase,row in enumerate(cases,1):
        require(int(row['phase']) == phase and int(row['timeout']) == CASES[phase][6] and
                int(row['code'],16) == CASES[phase][5] and int(row['data'],16) == CASES[phase][4] and
                int(row['t']) > by[phase]['report_ps'], 'case marker after retirement')
    return dict(status='PASS_LOCAL_MASTER_UNIT_PUBLIC_CONTRACT', transactions=completed,
                timeout_events=timeouts, timeout_limit_cycles=32, clock_period_ps=4000,
                configuration_releases_ps=reset_rises, channel_stall_cycles=stalls,
                native_xdma_tested=False, operational_reset_tested=False,
                scope='Ten black-box master transactions; native endpoint, CDC, inventory certificates and reset permission remain unqualified.')
