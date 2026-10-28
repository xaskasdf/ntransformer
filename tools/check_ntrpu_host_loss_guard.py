"""Review guard causality and retained payloads from three observed clock domains."""
import collections
import json
import re
import sys
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(line):
    out = {}
    for k, v in re.findall(r'(\w+)=(\S+)', line):
        out[k] = v if k == 'name' else int(v, 16 if k in {'qdata', 'qout', 'pdata', 'pout'} else 10)
    return out


def review(text, check_producer_holds=True):
    require(not re.search(r'^(Fatal:|ERROR:|FAIL )', text, re.M), 'failed simulation')
    require(text.splitlines().count('GUARD_SUITE_PASS cases=10') == 1, 'complete bank')
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in text.splitlines():
        if not line.startswith('GUARD_') or line.startswith('GUARD_SUITE_'):
            continue
        kind = line.split()[0]
        f = fields(line)
        groups[f['case']][kind].append(f)
    require(set(groups) == set(range(10)), 'ten unique cases')
    expected_cause = [0, 2, 1, 4, 4, 8, 2, 16, 2, 3]
    results = []
    for case, g in sorted(groups.items()):
        m, s, h = [g['GUARD_' + name] for name in ['M', 'S', 'H']]
        require(m and s and h, 'three observed clocks')
        require(len(g['GUARD_CASE_PASS']) == 1, 'single case verdict')
        for rows, period, half in [(m, 10000, 10000), (s, 5000, 2500), (h, 7300, 3650)]:
            for a, b in zip(rows, rows[1:]):
                dt = b['t'] - a['t']
                require(dt >= period and dt % half == 0, 'clock cadence or stopped-clock gap')
                if rows is m:
                    require(dt == period, 'management clock must remain continuous')
        q, p = collections.deque(), collections.deque()
        qr, qt, pr, pt, abandoned = [], [], [], [], []
        fence = False
        previous = None
        saw_invalid = False
        for row in s:
            if not row['cold']:
                q.clear(); p.clear(); qr.clear(); qt.clear(); pr.clear(); pt.clear(); abandoned.clear()
                fence = saw_invalid = False
                previous = None
                continue
            require(row['qfault'] == row['pfault'] == 0, 'channel contract fault')
            require(not saw_invalid or row['invalid'], 'invalidation must persist')
            saw_invalid |= bool(row['invalid'])
            require(row['qp'] == len(q) <= 2 and row['pp'] == len(p) <= 2, 'retained queue conservation')
            require((row['qa'], row['qd'], row['qb']) == (len(qr), len(qt), 0), 'request counters')
            require((row['pa'], row['pd'], row['pb']) == (len(pr), len(pt), len(abandoned)), 'response counters')
            require(row['pfence'] == int(fence), 'retained response fence')
            require(row['qo'] == int(bool(q)), 'requests remain exposed across interface reset')
            require(row['po'] == int(bool(p) and row['reset'] and not fence), 'response held until interface reset')
            if q:
                require(row['qout'] == q[0], 'request payload identity')
            if p:
                require(row['pout'] == p[0], 'response payload identity')
            if row['invalid'] or not row['reset']:
                require(row['qi'] == 0, 'no new request admission')
            if previous and check_producer_holds:
                if previous['qv'] and not previous['qi'] and previous['reset'] and row['reset']:
                    require(row['qv'] and row['qdata'] == previous['qdata'], 'request producer withdrew VALID before handshake/reset')
                if previous['pv'] and not previous['pi']:
                    require(row['pv'] and row['pdata'] == previous['pdata'], 'retained response producer withdrew VALID')
            if row['qo'] and row['qr']:
                qt.append(q.popleft())
            if row['po'] and row['pr']:
                pt.append(p.popleft())
            if row['qv'] and row['qi']:
                qr.append(row['qdata']); q.append(row['qdata'])
            if row['pv'] and row['pi']:
                pr.append(row['pdata']); p.append(row['pdata'])
            if fence or (row['invalid'] and not row['reset']):
                abandoned.extend(p); p.clear(); fence = True
            previous = row
        require(not q and not p and qr == qt == [0x12340, 0x12341], 'exact request fixture conservation')
        expected_responses = [0x055a0, 0x055a1] + ([0x066bb] if case else [])
        require(pr == expected_responses, 'response fixture identities')
        require((pt, abandoned) == ([], expected_responses) if case else (pt, abandoned) == (expected_responses, []), 'exact response disposition')
        cold_m = [r for r in m if r['cold']]
        running = [r for r in cold_m if r['running']]
        require(running and all(r['state'] == 3 and r['ready'] and not r['req'] for r in running), 'startup ready/clock gates')
        first_run = running[0]['t']
        require(all(not r['loss'] for r in cold_m if r['t'] < first_run), 'no loss latched before first session')
        require(sum(not r['link'] and r['ep'] for r in cold_m if r['t'] < first_run) >= 80, 'initial absent-link fixture')
        require(all(not r['ha'] and not r['sa'] for r in cold_m if r['t'] < first_run), 'startup admission closed')
        require(all(not r['req'] or (not r['hr'] and not r['sr']) for r in cold_m), 'both facing resets asserted together')
        events = {r['name']: r['t'] for r in g['GUARD_EVENT']}
        loss_rows = [r for r in cold_m if r['loss']]
        reset_spans = []
        loss_latency = invalidate_latency = None
        if case:
            require(loss_rows and loss_rows[-1]['causes'] == expected_cause[case], 'loss cause identity')
            first_loss = loss_rows[0]['t']
            require(all(r['loss'] and not r['running'] and not r['ha'] and not r['sa'] for r in cold_m if r['t'] >= first_loss), 'permanent admission closure')
            require(all(r['tick'] == loss_rows[0]['tick'] for r in loss_rows), 'first loss timestamp is retained')
            require(cold_m.index(loss_rows[0]) == loss_rows[0]['tick'], 'timestamp matches observed management cycles')
            for a, b in zip(loss_rows, loss_rows[1:]):
                require(b['causes'] | a['causes'] == b['causes'], 'cause history is retained')
            injection = min(t for name, t in events.items() if name.startswith(('inject_', 'stop_')))
            loss_latency = first_loss - injection
            require(0 <= loss_latency <= (800000 if case in {3, 4, 5} else 80000), 'bounded management detection')
            first_invalid = next(r['t'] for r in s if r['cold'] and r['invalid'])
            first_ack = next(r['t'] for r in s if r['cold'] and r['ack'])
            require(first_ack > first_invalid, 'ownership ACK must follow source invalidation')
            if case == 5:
                require(first_invalid > events['resume_source'], 'stopped source cannot acknowledge')
                invalidate_latency = first_invalid - events['resume_source']
            elif case == 7:
                invalidate_latency = first_invalid - injection
            else:
                invalidate_latency = first_invalid - first_loss
            require(0 <= invalidate_latency <= 50000, 'bounded source invalidation when clock available')
            start = None
            for a, b in zip(cold_m, cold_m[1:]):
                if b['t'] <= first_run:
                    continue
                if not a['req'] and b['req']:
                    require(b['t'] > first_loss and b['t'] - first_ack >= 20000, 'reset waits for returned ownership ACK')
                    start = b['t']
                if a['req'] and not b['req'] and start is not None:
                    require(b['held'] and b['halive'] and b['salive'] and b['link'] and not b['ep'], 'release prerequisites')
                    count_s = sum(start < r['t'] < b['t'] and not r['reset'] for r in s)
                    count_h = sum(start < r['t'] < b['t'] and not r['reset'] for r in h)
                    require(count_s >= 64 and count_h >= 64, 'both-domain reset overlap length')
                    reset_spans.append(dict(start_ps=start, end_ps=b['t'], source_edges=count_s, host_edges=count_h))
                    start = None
            require(start is None and len(reset_spans) == (2 if case == 9 else 1), 'all runtime reset spans completed')
            if case in {3, 4}:
                require(reset_spans[0]['start_ps'] < events['resume_host'] < reset_spans[0]['end_ps'], 'reset held through absent host clock')
            if case in {5, 6}:
                end = events['resume_source' if case == 5 else 'allow_ack']
                require(all(not r['req'] and r['hr'] and r['sr'] for r in cold_m if first_loss <= r['t'] < end), 'no reset without available core ACK')
            if case == 8:
                require(reset_spans[0]['end_ps'] > events['allow_peers_held'], 'peer reset acknowledgement required')
            require(cold_m[-1]['state'] == 8 and cold_m[-1]['hr'] and cold_m[-1]['sr'], 'restored interface remains session-invalid')
        else:
            require(not loss_rows and not saw_invalid, 'normal session stays valid')
        summary = g['GUARD_CASE_PASS'][0]
        require([summary[k] for k in ['causes', 'qa', 'qd', 'qb', 'pa', 'pd', 'pb']] ==
                [expected_cause[case], 2, 2, 0, 3 if case else 2, 0 if case else 2, 3 if case else 0], 'summary agrees with independent ledger')
        results.append(dict(case=case, management_samples=len(m), source_samples=len(s), host_samples=len(h),
                            cause=expected_cause[case], detection_ps=loss_latency, invalidation_ps=invalidate_latency,
                            reset_spans=reset_spans, request_captures=len(qr), response_captures=len(pr)))
    return dict(cases=results, producer_holds_checked=check_producer_holds,
                scope='Automatic guard/XPM probes and two retained channels; controlled ownership ACK and peer acknowledgements. No AXI Clock Converter, PCIe, NTRD ownership or DDR integration.')


if __name__ == '__main__':
    print(json.dumps(review(Path(sys.argv[1]).read_text(encoding='utf-8')), indent=2))
