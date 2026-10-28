"""Check reset/admission contracts using public pins and external clock counts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def review(text, profile):
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    need(profile in (0, 1, 2), 'profile')
    hp, dp = [(4000, 5000), (6200, 5000), (31000, 23000)][profile]
    need(not re.search(r'(?mi)^\s*(Fatal:|Error:|FAIL\b)', text), 'tool or bank error')
    configured = False
    cases, stopped, starts = [], [], Counter()
    previous = None
    asserted = {'h': 0, 'd': 0}
    release_edges, admission_edges = Counter(), Counter()
    last_time = -1
    finished = False
    sample_count = 0
    external_at = None
    resumed = {'h': True, 'd': True}
    stop_at, stop_latencies = None, []
    for line in text.splitlines():
        if not line.startswith('NTRPU_SUP_'):
            continue
        need(not finished, 'trace after PASS')
        kind = line.split()[0].removeprefix('NTRPU_SUP_')
        raw = dict(re.findall(r'(\w+)=([^\s]+)', line))
        need(all(re.fullmatch(r'\d+', value) for value in raw.values()), 'unknown or invalid pin')
        row = {key: int(value) for key, value in raw.items()}
        if kind == 'CONFIG':
            need(not configured, 'duplicate configuration')
            need(row == dict(profile=profile, host_ps=hp, device_ps=dp, mgmt_ps=10000,
                             hold=64, timeout=64, clear=8), 'configuration')
            configured = True
            continue
        need(configured, 'missing configuration')
        if kind == 'PASS':
            need(row == dict(cases=12, starts=17, profile=profile), 'PASS counts')
            need(cases == list(range(1, 13)), 'case coverage')
            need(stopped == [5, 6, 7, 8, 9], 'clock stop coverage')
            need(starts == Counter({i: 2 if i in {4, 5, 6, 7, 10} else 1
                                    for i in range(1, 13)}), 'restart coverage')
            need(previous and previous['run'] and previous['hadmit'] and previous['dadmit'], 'not running at end')
            finished = True
            continue
        need(row['ps'] >= last_time, 'time order')
        last_time = row['ps']
        if kind == 'CASE':
            need(row['id'] == len(cases) + 1, 'case order')
            cases.append(row['id'])
            continue
        need(row['case'] == cases[-1], 'case identity')
        if kind == 'STOPPED':
            expected = {5: (1, 0), 6: (0, 1), 7: (1, 1), 8: (1, 0), 9: (0, 1)}
            need(row['case'] in expected and (row['host'], row['device']) == expected[row['case']], 'stopped domain')
            need(previous and previous['req'] and not any(previous[k] for k in ['hrst', 'drst', 'hadmit', 'dadmit', 'can']), 'stopped clock not blocked')
            stopped.append(row['case'])
            continue
        if kind == 'STARTED':
            starts[row['case']] += 1
            need(row['count'] == sum(starts.values()), 'start count')
            need(previous and previous['run'] and previous['hadmit'] and previous['dadmit'], 'start before admission')
            continue
        need(kind in {'M', 'OUT', 'STIM'}, 'unknown marker')
        required = {'case', 'ps', 'ht', 'dt', 'mt', 'he', 'de', 'me', 'ext', 'held', 'ready',
                    'rearm', 'hen', 'den', 'req', 'hrst', 'drst', 'hadmit', 'dadmit',
                    'can', 'halive', 'dalive', 'run'}
        need(set(row) == required, 'sample fields')
        need(all(row[k] in (0, 1) for k in required - {'case', 'ps', 'ht', 'dt', 'mt', 'he', 'de', 'me'}), 'nonbinary pin')
        sample_count += 1
        if kind == 'M':
            need(row['ps'] == 5000 + (row['mt'] - 1) * 10000 + 1, 'management clock accounting')
        if row['ext'] and (previous is None or not previous['ext']):
            external_at = row['ps']
            asserted = {'h': row['ht'], 'd': row['dt']}
        if previous:
            if kind == 'STIM' and ((previous['hen'] and not row['hen']) or
                                   (previous['den'] and not row['den'])):
                need(stop_at is None, 'overlapping clock stop record')
                stop_at = row['ps']
            for prefix, period in [('h', hp), ('d', dp), ('m', 10000)]:
                tick, edge = prefix + 't', prefix + 'e'
                need(row[tick] >= previous[tick] and row[edge] >= previous[edge], 'clock count reversed')
                if prefix in {'h', 'd'}:
                    enabled = prefix + 'en'
                    if row[enabled] and not previous[enabled]:
                        resumed[prefix] = True
                    if not previous[enabled] and not row[enabled]:
                        need(row[tick] == previous[tick], 'edges while clock stopped')
                    if previous[enabled] and row[enabled] and previous[edge] and row[tick] > previous[tick]:
                        if not resumed[prefix]:
                            need(row[edge] - previous[edge] == period * (row[tick] - previous[tick]), 'clock period')
                        resumed[prefix] = False
            for prefix in ['h', 'd']:
                reset, tick, edge, admit = prefix + 'rst', prefix + 't', prefix + 'e', prefix + 'admit'
                if previous[reset] and not row[reset]:
                    asserted[prefix] = row[tick]
                if not previous[reset] and row[reset]:
                    need(row['ps'] == row[edge] + 1, 'asynchronous reset release')
                    need(row[tick] - asserted[prefix] >= 64, 'reset shorter than 64 real edges')
                    need(not row['ext'] and not row['req'], 'reset released during hold')
                    release_edges[prefix] += 1
                if not previous[admit] and row[admit]:
                    need(row['ps'] == row[edge] + 1, 'asynchronous admission release')
                    admission_edges[prefix] += 1
        # STIM is recorded before combinational/asynchronous propagation settles.
        if kind != 'STIM':
            if stop_at is not None and row['req'] and not any(row[k] for k in ['hrst', 'drst', 'hadmit', 'dadmit']):
                delay = row['ps'] - stop_at
                need(0 <= delay <= 800000, 'clock-loss reset deadline')
                stop_latencies.append(delay)
                stop_at = None
            if row['ext'] and external_at is not None and row['ps'] > external_at:
                need(row['req'] and not any(row[k] for k in ['hrst', 'drst', 'hadmit', 'dadmit']), 'external reset response')
            if row['hadmit'] or row['dadmit']:
                need(row['run'] and row['hrst'] and row['drst'] and row['ready'] and not row['req'] and not row['ext'], 'unsafe admission')
            if row['can']:
                need(row['held'] and row['halive'] and row['dalive'] and row['req'] and not row['ext'], 'unsafe rearm')
        previous = row
    need(finished and sample_count > 1000, 'incomplete trace')
    need(admission_edges == {'h': 17, 'd': 17}, 'admission edge count')
    need(len(stop_latencies) == 5 and stop_at is None, 'clock-loss latency coverage')
    return dict(profile=profile, cases=12, starts=17, clock_stop_cases=5, samples=sample_count,
                reset_releases=dict(release_edges), admission_releases=dict(admission_edges),
                stopped_clock_to_block_ps=stop_latencies)


def mutations(text, profile):
    changes = {
        'missing_pass': text.replace('NTRPU_SUP_PASS', 'REMOVED_PASS', 1),
        'missing_case': text.replace('NTRPU_SUP_CASE id=8 ', 'REMOVED_CASE id=8 ', 1),
        'missing_clock_stop': text.replace('NTRPU_SUP_STOPPED case=5 ', 'REMOVED_STOP case=5 ', 1),
        'wrong_hold_contract': text.replace('hold=64', 'hold=16', 1),
        'unknown_pin': re.sub(r'(NTRPU_SUP_M [^\n]*? hrst=)[01]', r'\g<1>x', text, count=1),
        'tool_error': text + '\nError: injected evidence failure\n',
    }
    # Perturb a real release edge, not an arbitrary inactive sample.
    lines = text.splitlines()
    previous = 0
    for index, line in enumerate(lines):
        if not re.match(r'NTRPU_SUP_(M|OUT|STIM) ', line):
            continue
        value = int(re.search(r' hrst=(\d)', line)[1])
        if value and not previous:
            changed = list(lines)
            changed[index] = re.sub(r' he=(\d+)', lambda m: ' he=' + str(int(m[1]) - 1), line)
            changes['off_edge_release'] = '\n'.join(changed)
            break
        previous = value
    rejected = []
    for name, changed in changes.items():
        assert changed != text
        try:
            review(changed, profile)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Accepted corrupted evidence: ' + name)
    assert len(rejected) == 7
    return rejected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--profile', type=int, required=True)
    args = parser.parse_args()
    text = args.log.read_text()
    print(json.dumps(dict(status='PASS', **review(text, args.profile),
                          rejected_mutations=mutations(text, args.profile))))
