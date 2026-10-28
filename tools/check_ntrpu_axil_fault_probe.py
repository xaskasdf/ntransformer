"""Review the positive prefix separately from deliberate post-certificate faults."""
import collections
import re

from check_ntrpu_axil_retained_fence import review as review_prefix


def review(text, period, expect_revoke=0):
    def need(ok, message):
        if not ok:
            raise ValueError(message)

    def fields(line):
        return dict(re.findall(r'(\w+)=(\S+)', line))

    need(expect_revoke in [0, 1], 'expected contract')
    marker = (f'FAULT_SUITE_PASS cases=3 device_period_ps={period} '
              f'expect_revoke={expect_revoke} native_endpoint=0 complete_dma_drain=0')
    need(text.splitlines().count(marker) == 1, 'fault suite scope')
    need(not re.search(r'^\s*(?:Fatal:|ERROR:|CRITICAL WARNING:|FAIL )', text, re.M), 'simulator failure')
    events = collections.defaultdict(list)
    passes = {}
    for line in text.splitlines():
        if line.startswith('FAULT_EVENT '):
            row = fields(line)
            name = row.pop('name')
            row = {k: int(v) for k, v in row.items()}
            row['name'] = name
            events[row['case']].append(row)
        if line.startswith('FAULT_CASE_PASS '):
            row = {k: int(v) for k, v in fields(line).items()}
            need(row['case'] not in passes, 'duplicate fault pass')
            passes[row['case']] = row
    need(set(events) == set(passes) == set(range(3)), 'three fault scenarios')
    expected_names = ['BASELINE_CERTIFIED', 'SOURCE_FAULT_CLOCK_STOPPED',
                      'AFTER_PROPAGATION', 'FRESH_ID_BLOCKED']
    if expect_revoke:
        expected_names.append('RETAINED_AFTER_SOURCE_CLEAR')
    expected_names.append('FAULT_TERMINAL')
    results = []
    for case in range(3):
        rows = events[case]
        need([r['name'] for r in rows] == expected_names, 'ordered fault observations')
        need(all(a['ps'] < b['ps'] for a, b in zip(rows, rows[1:])), 'fault chronology')
        baseline, injected, propagated, fresh = rows[:4]
        need(baseline['ack'] == 1 and baseline['first_ps'] == baseline['revoked_ps'] == 0,
             'certificate before fault')
        need(all(baseline[k] == 0 for k in ['gate_fault', 'h_fault', 'd_fault', 'fault_mgmt']),
             'clean baseline')
        source = ['gate_fault', 'h_fault', 'd_fault'][case]
        need(injected[source] == 1, 'actual selected ledger/admission fault')
        need(injected['device_run' if case == 2 else 'host_run'] == 0, 'faulting clock stopped')
        first = injected['first_ps']
        need(baseline['ps'] < first <= injected['ps'], 'source fault timestamp')
        need(propagated['ps'] - injected['ps'] >= 150000, 'management propagation window')
        need(all(r['first_ps'] == first for r in rows[1:]), 'stable origin timestamp')
        need(all(r['closed'] == r['hn'] == r['dn'] == 1 for r in rows), 'no reset/reopening')
        need(fresh['ack'] == rows[-1]['ack'] == 0, 'fresh identity remains blocked')
        need(rows[-1]['host_run'] == rows[-1]['device_run'] == 1, 'clocks resumed')
        if expect_revoke:
            revoked = propagated['revoked_ps']
            need(first <= revoked <= first + 30000 and revoked % 10000 == 5000,
                 'revocation on management edge within three periods')
            need(all(r['ack'] == 0 and r['fault_mgmt'] & (1 << case) for r in rows[2:]),
                 'selected fault retained and certificate revoked')
            need(all(r['revoked_ps'] == revoked for r in rows[2:]), 'stable revocation timestamp')
            cleared = rows[-2]
            need(all(cleared[k] == 0 for k in ['gate_fault', 'h_fault', 'd_fault']),
                 'source reports really removed')
            need(cleared['fault_mgmt'] == rows[-1]['fault_mgmt'], 'management retention')
        else:
            revoked = 0
            need(propagated['ack'] == 1 and propagated[source] == 1, 'stale certificate counterexample')
            need(all(r['fault_mgmt'] == r['revoked_ps'] == 0 for r in rows), 'original controller observation')
        p = passes[case]
        need(p == dict(case=case, expect_revoke=expect_revoke, first_ps=first,
                       revoked_ps=revoked, fault_mgmt=rows[-1]['fault_mgmt']), 'fault pass measurements')
        results.append(dict(case=case, source=source, first_ps=first, revoked_ps=revoked,
                            latency_ps=revoked-first if expect_revoke else None, events=rows))
    # Only the unchanged pre-fault interval belongs to the conservation regression.
    prefix_lines = []
    for line in text.splitlines():
        if line.startswith('FENCE_BUS '):
            row = fields(line)
            if int(row['ps']) >= events[int(row['case'])][0]['ps']:
                continue
        prefix_lines.append(line)
    prefix_lines.append(f'FENCE_SUITE_PASS cases=3 device_period_ps={period} native_endpoint=0 complete_dma_drain=0')
    prefix = review_prefix('\n'.join(prefix_lines), period)
    return dict(status='PASS_FAULT_REVOCATION' if expect_revoke else 'CONFIRMED_STALE_SECTION_CERTIFICATE',
                device_period_ps=period, positive_prefix=prefix, fault_cases=results,
                qualified_native_endpoint=False, complete_dma_drain=False)
