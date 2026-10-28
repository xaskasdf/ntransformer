"""Reconstruct warm-up traffic, injected orphan responses and lifetime events."""
import collections
import re


def review(text, period, stop_source):
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    def fields(line):
        return dict(re.findall(r'(\w+)=(\S+)', line))

    need(period in [5000, 37000] and stop_source in [0, 1], 'clock profile')
    need(not re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:|FAIL )', text, re.M), 'simulator failure')
    marker = (f'LIFE_SUITE_PASS cases=4 device_period_ps={period} stop_source={stop_source} '
              'native_endpoint=0 complete_dma_drain=0')
    need(text.splitlines().count(marker) == 1 and '$finish called at time' in text, 'complete scoped suite')
    events = collections.defaultdict(list)
    bus = collections.defaultdict(list)
    resets = collections.defaultdict(list)
    passes = {}
    for line in text.splitlines():
        if line.startswith('LIFE_EVENT '):
            row = fields(line)
            name = row.pop('name')
            row = {k: int(v, 16 if k == 'rdata' else 10) for k, v in row.items()}
            row['name'] = name
            events[row['case']].append(row)
        elif line.startswith('LIFE_CASE_PASS '):
            row = {k: int(v) for k, v in fields(line).items()}
            need(row['case'] not in passes, 'unique pass')
            passes[row['case']] = row
        elif line.startswith('FENCE_BUS '):
            row = fields(line)
            case, side, channel = int(row.pop('case')), row.pop('side'), row.pop('channel')
            need(case in range(4) and side in ['U', 'G', 'H', 'D'], 'bus identity')
            need(channel in ['AW', 'W', 'B', 'AR', 'R'], 'bus channel')
            row = {k: int(v, 10 if k in ['ps', 'close'] else 16) for k, v in row.items()}
            step, phase = (period // 2, 1100) if side == 'D' else (3500, 700)
            need(row['ps'] % step == phase, 'gated clock grid')
            bus[case, side, channel].append(row)
        elif line.startswith('AUTO_RESET '):
            row = fields(line)
            resets[int(row['case'])].append(row)
    need(set(events) == set(passes) == set(range(4)), 'four complete scenarios')
    names = ['START', 'ARMED', 'SOURCE_FAULT', 'PROPAGATED', 'FRESH_BLOCKED',
             'SOURCE_REPORTS_CLEARED', 'TERMINAL']
    results = []
    for case in range(4):
        profile = (f'LIFE_PROFILE case={case} device_period_ps={period} '
                   f'host_period_ps=7000 mgmt_period_ps=10000 stop_source={stop_source}')
        need(text.splitlines().count(profile) == 1, 'individual profile')
        ev = events[case]
        need([r['name'] for r in ev] == names, 'ordered lifecycle')
        need(all(a['ps'] < b['ps'] for a, b in zip(ev, ev[1:])), 'lifecycle chronology')
        start, armed, injected, propagated, fresh, cleared, terminal = ev
        need(start['req'] == start['ack'] == start['close'] == start['state'] == 0, 'healthy open baseline')
        need(all(start[k] == 0 for k in ['gate_sources', 'gate_fault', 'h_fault', 'd_fault',
                                       'fault_mgmt', 'first_ps', 'report_ps', 'close_ps']), 'no initial fault')
        first, report, close = terminal['first_ps'], terminal['report_ps'], terminal['close_ps']
        need(armed['ps'] < first <= injected['ps'], 'fault really follows arming')
        need(0 <= report - first <= 30000 and report % 10000 == 5000, 'management latency')
        expected_fault = 1 if case in [0, 2] else 12
        need(all(r['first_ps'] == first for r in ev[2:]), 'stable fault origin')
        need(all(r['report_ps'] == report and r['fault_mgmt'] == (4 if case in [1, 3] and (case == 1 or stop_source == 1) and r['name'] in ['PROPAGATED', 'FRESH_BLOCKED'] else expected_fault) and r['ack'] == 0
                 for r in ev[3:]), 'retained management fault and revoked authorization')
        need(all(r['hn'] == r['dn'] == 1 for r in ev), 'reset rails remain released')
        expected_sources = 1 if case == 0 else 2 if case == 2 else 0
        need(injected['gate_sources'] == expected_sources, 'registered admission fault identity')
        need(injected['gate_fault'] == (1 if case in [0, 2] else 0), 'admission fault detector')
        need(injected['d_fault'] == (1 if case in [1, 3] else 0), 'device fault detector')
        for row in ev[2:5]:
            need(row['host_run'] == (0 if case == 1 else 1 - stop_source if case in [0, 2] else 1),
                 'observed host clock state')
            need(row['device_run'] == (1 - stop_source if case in [1, 3] else 1), 'device clock state')
        need(fresh['req'] == 1 and fresh['generation'] == armed['generation'] + 1
             and fresh['tag'] == armed['tag'] + 1, 'fresh request identity exercised')
        need(cleared['gate_sources'] == cleared['h_fault'] == cleared['d_fault'] == 0,
             'all actual fault inputs removed')
        need(cleared['guard_sources'] == 0, 'actual guard CDC sources removed')
        need(all(r['host_run'] == r['device_run'] == r['close'] == 1 for r in ev[-2:]),
             'clocks resumed with permanent closure')
        if case == 1:
            need(armed['req'] == armed['send'] == armed['busy'] == 1 and armed['ack'] == armed['close'] == 0,
                 'outstanding mailbox armed with stopped peer')
            for row in ev[1:5]:
                need(row['send'] == row['busy'] == 1 and row['received'] == row['close'] == 0,
                     'no mailbox abort while peer clock stopped')
                need(row['source_tag'] == armed['tag'] and row['source_generation'] == armed['generation'],
                     'mailbox payload remains old identity')
            need(close > fresh['ps'], 'host closure waits for host clock')
            need(terminal['source_tag'] == armed['tag'] and terminal['source_generation'] == armed['generation'],
                 'no relabel or fresh mailbox issue')
            need((terminal['busy'], terminal['send'], terminal['obsolete']) in [(1, 1, 0), (0, 0, 1)],
                 'old mailbox waits or completes obsolete')
        elif case == 2:
            for row in ev[1:5]:
                need(row['close'] == row['bvalid'] == row['rvalid'] == 1
                     and row['bresp'] == row['rresp'] == 2 and row['rdata'] == 0,
                     'local SLVERR payload retained across fault and clock stop')
                need(row['ua'] == row['uw'] == row['uq'] == 1, 'held local obligations retained')
            need(armed['ack'] == 1 and close == 0, 'already certified before held-error fault')
        else:
            need(armed['req'] == armed['ack'] == armed['close'] == 0, 'fault before any proof')
            need(first <= close <= first + (65000 if case == 3 else 1), 'causal fault-only closure latency')
        need(terminal['ua'] == terminal['uw'] == terminal['uq'] == 0, 'no remaining upstream obligations')
        need(terminal['bvalid'] == terminal['rvalid'] == 0, 'upstream responses completed')
        address = 0x7000 + case * 4
        datum = 0xa1b2c3d4 + case
        old = 0x13579bdf ^ (((0x1c00 + case) * 0x01020408) & 0xffffffff)
        readback = (old & 0xff00ff00) | (datum & 0x00ff00ff)
        expected = {'AW': [(address,)], 'W': [(datum, 5)], 'B': [(case,)],
                    'AR': [(address,)], 'R': [(readback, case)]}
        keys = {'AW': ['addr'], 'W': ['data', 'strb'], 'B': ['resp'], 'AR': ['addr'], 'R': ['data', 'resp']}
        for side in ['U', 'G', 'H', 'D']:
            for channel, fields_to_compare in keys.items():
                want = list(expected[channel])
                if case == 2 and side == 'U':
                    want += {'AW': [(0x7800,)], 'W': [(0xdeadbeef, 9)], 'B': [(2,)],
                             'AR': [(0x7800,)], 'R': [(0, 2)]}[channel]
                if case == 0 and side in ['U', 'G'] and channel == 'R':
                    want.append((0xbad00bad, 2))
                if case in [1, 3] and side == 'D' and channel == 'B':
                    want.append((case,))
                actual = [tuple(r[k] for k in fields_to_compare) for r in bus[case, side, channel]]
                need(actual == want, f'payload/response inventory {case}/{side}/{channel}')
        counts = dict(uaw=len(bus[case, 'U', 'AW']), uar=len(bus[case, 'U', 'AR']),
                      ubc=len(bus[case, 'U', 'B']), urc=len(bus[case, 'U', 'R']),
                      gaw=len(bus[case, 'G', 'AW']), gar=len(bus[case, 'G', 'AR']),
                      gbc=len(bus[case, 'G', 'B']), grc=len(bus[case, 'G', 'R']),
                      daw=len(bus[case, 'D', 'AW']), dar=len(bus[case, 'D', 'AR']),
                      dbc=len(bus[case, 'D', 'B']), drc=len(bus[case, 'D', 'R']))
        need(passes[case] == dict(case=case, first_ps=first, report_ps=report, close_ps=close,
                                 fault_mgmt=expected_fault, **counts), 'reported counts independently reconstructed')
        rr = resets[case]
        need([r['event'] for r in rr] == ['START', 'HELD', 'READY'], 'configuration reset only')
        need(int(rr[1]['he']) >= 64 and int(rr[1]['de']) >= 64 and int(rr[-1]['ps']) < start['ps'], 'startup reset complete')
        raw = [fields(line) for line in text.splitlines() if line.startswith('GUARD_RAW ')
               and int(fields(line)['case']) == case]
        need(len(raw) == (3 if case in [1, 3] else 2), 'exact raw response offers')
        normal_b = next(r for r in raw if r['channel'] == 'B')
        normal_r = next(r for r in raw if r['channel'] == 'R')
        need(int(normal_b['aw']) == int(normal_b['w']) == int(normal_r['ar']) == 1,
             'normal responses have prior request obligations')
        if case in [1, 3]:
            orphan = [r for r in raw if r['channel'] == 'B'][1]
            need(int(orphan['ps']) >= first and orphan['ready'] == orphan['protected_valid'] == '0'
                 and orphan['aw'] == orphan['w'] == '0', 'causal orphan quarantine')
            need(terminal['guard_sources'] == 1 and terminal['raw_bvalid'] == 1
                 and terminal['raw_bready'] == terminal['protected_bvalid'] == 0,
                 'orphan remains held without manager acceptance')
        else:
            need(terminal['guard_sources'] == 0, 'no spurious guard fault')
        results.append(dict(case=case, raw_response_offers=raw, report_latency_ps=report-first,
                            closure_latency_ps=close-first if case in [0, 3] else None,
                            counters=counts, events=ev))
    return dict(status='PASS_FOUR_SECTION_LIFETIME_CASES', period_ps=period,
                stop_source=stop_source, cases=results, native_endpoint=False, complete_dma_drain=False)
