"""Read-only, supplemental review of the archived DDR R1 capture.

This does not replace REVIEW.json, claim native execution, or authorize power.
SAIF occupancy cannot recover correlated commands or per-rank transactions.
"""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


SECONDS = {unit: Decimal(value) for unit, value in
           [('s', '1'), ('ms', '.001'), ('us', '.000001'),
            ('ns', '1e-9'), ('ps', '1e-12'), ('fs', '1e-15')]}
WINDOW_SECONDS = Decimal('0.000020480')
CAPTURE_SOURCE_SHA256 = 'acb1a3923b940cc5d9d076c264c30787336ed188b7eaf97e837291e0fc2442a3'
METRICS = ('cycles', 'sample_ps', 'aw_count', 'b_count', 'ar_count',
           'rlast_count', 'w_beats', 'r_beats', 'w_bytes', 'w_stalls',
           'r_stalls', 'pending_w', 'pending_r', 'simulator_time')
STEADY = {
    'sys_rst': 0, 'ntrpu_mig_reset': 0, 'c0_ddr4_rst': 0,
    'c0_ddr4_aresetn': 1, 'c0_init_calib_complete': 1,
    'c0_ddr4_reset_n': 1, 'c0_data_compare_error': 0,
    'c0_ddr4_write_err': 0, 'c0_ddr4_read_err': 0,
    'u_ddr4_v2_2_30_axi_tg_top/vio_axi_tg_prbs_mode_running': 1,
}
FIELDS = {'T0', 'T1', 'TX', 'TZ', 'TC', 'IG', 'TB'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream)


def _digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(chunk)
    return value.hexdigest()


def tokens(stream):
    """Bounded lexer; preserve escaped delimiters and quoted header strings."""
    token, quoted, escaped = [], False, False
    for chunk in iter(lambda: stream.read(65536), ''):
        for char in chunk:
            if escaped:
                token.append(char)
                escaped = False
            elif char == '\\':
                token.append(char)
                escaped = True
            elif char == '"':
                quoted = not quoted
                token.append(char)
            elif quoted:
                token.append(char)
            elif char.isspace() or char in '()':
                if token:
                    yield ''.join(token)
                    token = []
                if char in '()':
                    yield char
            else:
                token.append(char)
            require(len(token) <= 65536, 'Oversized SAIF token')
    require(not quoted and not escaped, 'Unterminated SAIF string/escape')
    if token:
        yield ''.join(token)


def identifier(raw):
    if raw.startswith('"'):
        require(raw.endswith('"'), 'Unterminated identifier')
        raw = raw[1:-1]
    return re.sub(r'\\(.)', r'\1', raw)


def parse_saif(path):
    """Parse the simple backward SAIF emitted by XSim; reject unknown forms."""
    path = Path(path)
    require(0 < path.stat().st_size <= 512 * 1024 * 1024, 'SAIF size outside review limit')
    stack, headers, signals, roots, instances = [], {}, {}, 0, set()
    with path.open(encoding='utf-8') as stream:
        for token in tokens(stream):
            if token == '(':
                if not stack:
                    roots += 1
                    require(roots == 1, 'Multiple SAIF roots')
                require(len(stack) < 256, 'SAIF hierarchy too deep')
                stack.append(dict(atoms=[], fields={}, children=0))
                continue
            require(stack, 'Atom outside SAIF expression')
            if token != ')':
                require(not stack[-1]['children'], 'Atom after SAIF child expression')
                stack[-1]['atoms'].append(token)
                continue
            row = stack.pop()
            atoms, fields = row['atoms'], row['fields']
            require(bool(atoms), 'Empty SAIF expression')
            key = atoms[0]
            parent = stack[-1]['atoms'][0] if stack and stack[-1]['atoms'] else None
            if not stack:
                require(key == 'SAIFILE' and len(atoms) == 1, 'Expected SAIFILE root')
            elif parent == 'SAIFILE' and key != 'INSTANCE':
                require(key not in headers and not row['children'], 'Duplicate/nested SAIF header')
                headers[key] = [identifier(a) for a in atoms[1:]]
            elif key == 'INSTANCE':
                require(parent in ('SAIFILE', 'INSTANCE') and len(atoms) == 2, 'Bad INSTANCE')
                scope = tuple(identifier(s['atoms'][1]) for s in stack
                              if s['atoms'][:1] == ['INSTANCE']) + (identifier(atoms[1]),)
                require(scope not in instances, 'Duplicate instance scope')
                instances.add(scope)
            elif key in ('NET', 'PORT') and parent == 'INSTANCE':
                require(len(atoms) == 1 and not fields, 'Bad signal container')
            elif parent in ('NET', 'PORT'):
                require(len(atoms) == 1 and {'T0', 'T1', 'TX', 'TC'} <= fields.keys(), 'Incomplete signal counters')
                scope = [identifier(s['atoms'][1]) for s in stack if s['atoms'][:1] == ['INSTANCE']]
                if scope[:1] == ['ntrpu_ddr_example_activity_tb']:
                    scope = scope[1:]
                require(scope[:1] == ['dut'], 'Signal outside archived DUT')
                name = '/'.join(scope + [identifier(key)])
                require(name not in signals, 'Duplicate canonical signal: ' + name)
                signals[name] = {f: fields.get(f, 0) for f in sorted(FIELDS)}
            elif key in FIELDS and len(stack) >= 2 and stack[-2]['atoms'][0] in ('NET', 'PORT'):
                require(len(atoms) == 2 and not row['children'], 'Bad signal counter')
                require(re.fullmatch(r'[0-9]+', atoms[1]) is not None, 'Nonnegative integer counter required')
                require(key not in stack[-1]['fields'], 'Duplicate signal counter')
                stack[-1]['fields'][key] = int(atoms[1])
            else:
                raise ValueError('Unsupported SAIF expression: ' + str(atoms[:2]))
            if stack:
                stack[-1]['children'] += 1
    require(not stack and roots == 1 and bool(signals), 'Incomplete SAIF')
    require(headers.get('SAIFVERSION') == ['2.0'], 'Unsupported SAIF version')
    require(headers.get('DIRECTION', ['backward']) == ['backward'], 'Expected backward SAIF')
    require(headers.get('DIVIDER', ['/']) == ['/'], 'Unsupported hierarchy divider')
    require(len(headers.get('DURATION', [])) == 1 and len(headers.get('TIMESCALE', [])) == 2,
            'Missing duration/timescale')
    duration = int(headers['DURATION'][0])
    factor, unit = headers['TIMESCALE']
    require(unit in SECONDS, 'Unsupported time unit')
    scale = Decimal(factor)
    require(scale.is_finite() and scale > 0 and duration > 0, 'Invalid timescale/duration')
    require(duration * scale * SECONDS[unit] == WINDOW_SECONDS, 'Unexpected SAIF duration')
    for name, values in signals.items():
        require(sum(values[k] for k in ('T0', 'T1', 'TX', 'TZ')) == duration,
                'Residency mismatch: ' + name)
    require(sum(s['TC'] for s in signals.values()) > 0, 'No transitions')
    return dict(sha256=digest(path), bytes=path.stat().st_size, headers=headers,
                duration_ticks=duration, duration_seconds=str(WINDOW_SECONDS), signals=signals)


def simulation_seconds(text):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(s|ms|us|ns|ps|fs)', text)
    require(match is not None, 'Malformed simulator time: ' + text)
    return Decimal(match[1]) * SECONDS[match[2]]


def review_windows(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(reader.fieldnames == ['window', 'edge', 'metric', 'value'], 'Unexpected window columns')
        values = {}
        for row in reader:
            require(None not in row and None not in row.values(), 'Malformed window row')
            key = row['window'], row['edge'], row['metric']
            require(key not in values, 'Duplicate window metric')
            values[key] = row['value']
    expected = {('all', 'scope', 'objects_selected')}
    expected |= {(f'prbs_{i}', edge, metric) for i in range(2)
                 for edge in ('start', 'end') for metric in METRICS}
    require(set(values) == expected, 'Missing or extra window metrics')
    require(int(values['all', 'scope', 'objects_selected']) > 0, 'No selected objects')
    windows = []
    for i in range(2):
        edges = {}
        for edge in ('start', 'end'):
            name = f'prbs_{i}'
            row = {metric: int(values[name, edge, metric]) for metric in METRICS if metric != 'simulator_time'}
            require(all(v >= 0 for v in row.values()), 'Negative counter')
            require(row['aw_count'] - row['b_count'] == row['pending_w'], 'Write accounting')
            require(row['ar_count'] - row['rlast_count'] == row['pending_r'], 'Read accounting')
            row['simulator_seconds'] = str(simulation_seconds(values[name, edge, 'simulator_time']))
            # sample_ps is the last UI rising edge, never in the future or >= one period old.
            age = Decimal(row['simulator_seconds']) - Decimal(row['sample_ps']) * SECONDS['ps']
            require(0 <= age < Decimal('5e-9'), 'Stale/future UI sample')
            edges[edge] = row
        start, end = edges['start'], edges['end']
        require(Decimal(end['simulator_seconds']) - Decimal(start['simulator_seconds']) == WINDOW_SECONDS,
                'Simulator interval differs from SAIF interval')
        delta = {k: end[k] - start[k] for k in start if k != 'simulator_seconds'}
        require(delta['cycles'] == 4096 and delta['sample_ps'] == 20480000, 'Window size')
        require(all(v >= 0 for k, v in delta.items() if not k.startswith('pending_')), 'Counters decreased')
        for metric in ('aw_count', 'b_count', 'ar_count', 'rlast_count', 'w_beats', 'r_beats', 'w_stalls', 'r_stalls'):
            require(delta[metric] <= 4096, 'More than one event per UI cycle: ' + metric)
        require(delta['w_beats'] + delta['w_stalls'] <= 4096, 'Overlapping accepted/stalled writes')
        require(delta['r_beats'] + delta['r_stalls'] <= 4096, 'Overlapping accepted/stalled reads')
        require(delta['rlast_count'] <= delta['r_beats'], 'RLAST without accepted R beat')
        require(0 <= delta['w_bytes'] <= 64 * delta['w_beats'], 'Invalid write byte count')
        require(delta['w_beats'] + delta['r_beats'] > 0, 'No accepted data')
        if windows:
            require(windows[-1]['end'] == start, 'Noncontiguous windows')
        rates = {
            'accepted_write_bytes_per_second': str(Decimal(delta['w_bytes']) / WINDOW_SECONDS),
            'accepted_read_bytes_per_second': str(Decimal(delta['r_beats'] * 64) / WINDOW_SECONDS),
            'write_stall_ui_fraction': str(Decimal(delta['w_stalls']) / 4096),
            'read_stall_ui_fraction': str(Decimal(delta['r_stalls']) / 4096),
        }
        windows.append(dict(name=name, **edges, delta=delta, rates=rates))
    require(sum(w['delta']['w_beats'] for w in windows) > 0 and
            sum(w['delta']['r_beats'] for w in windows) > 0, 'Missing read or write traffic')
    return dict(objects_selected=int(values['all', 'scope', 'objects_selected']), windows=windows)


def summarize_saif(parsed):
    signals, duration = parsed['signals'], parsed['duration_ticks']
    issues, checks = [], {}
    for suffix, expected in STEADY.items():
        name = 'dut/' + suffix
        row = signals.get(name)
        if row is None:
            checks[name] = dict(status='MISSING', expected=expected)
            issues.append('Missing critical signal: ' + name)
            continue
        good = row['T' + str(expected)] == duration and all(row[k] == 0 for k in ('TC', 'IG', 'TB'))
        checks[name] = dict(status='STEADY' if good else 'NOT_STEADY', expected=expected, counters=row)
        if not good:
            issues.append('Critical signal not steadily healthy: ' + name)
    clock = signals.get('dut/c0_ddr4_clk')
    clock_good = (clock is not None and clock['TC'] == 8192 and clock['T0'] * 2 == duration
                  and clock['T1'] * 2 == duration and clock['TB'] == 0 and clock['IG'] == 0)
    if not clock_good:
        issues.append('UI clock missing or not 4096 complete 50% duty cycles')
    groups = defaultdict(lambda: dict(signals=0, transitions=0, active_signals=0,
                                      x_ticks=0, z_ticks=0, tb_ticks=0, ig_count=0))
    for name, row in signals.items():
        parts = name.split('/')
        group = '/'.join(parts[:2]) if len(parts) > 2 else 'dut/<top-signals>'
        g = groups[group]
        g['signals'] += 1
        g['transitions'] += row['TC']
        g['active_signals'] += int(row['TC'] > 0)
        for dest, key in [('x_ticks', 'TX'), ('z_ticks', 'TZ'), ('tb_ticks', 'TB'), ('ig_count', 'IG')]:
            g[dest] += row[key]
    for group in groups.values():
        group['xz_signal_time_fraction'] = str(Decimal(group['x_ticks'] + group['z_ticks']) / (duration * group['signals']))
    unknown = sorted((dict(signal=n, counters=r) for n, r in signals.items() if r['TX'] or r['TZ'] or r['TB']),
                     key=lambda v: (-(v['counters']['TX'] + v['counters']['TZ'] + v['counters']['TB']), v['signal']))
    pins = []
    for rank in range(4):
        # Exact pin occupancy only. A low CS without clock/CKE/command correlation
        # is NOT evidence that the corresponding DRAM rank accepted a command.
        observations = {p: signals.get(f'dut/c0_ddr4_{p}[{rank}]') for p in ('cs_n', 'cke', 'odt')}
        pins.append(dict(rank_index=rank, slot=rank // 2, rank_in_slot=rank % 2,
                         pin_counters=observations, commands_observed=None, refresh_observed=None))
    return dict(sha256=parsed['sha256'], bytes=parsed['bytes'], duration_ticks=duration,
                headers=parsed['headers'], duration_seconds=str(WINDOW_SECONDS), signals=len(signals),
                signal_names_sha256=hashlib.sha256('\n'.join(sorted(signals)).encode()).hexdigest(),
                critical_signals=checks, ui_clock=dict(valid_aggregate=clock_good, counters=clock),
                groups=dict(sorted(groups.items())), signals_with_x_z_or_tb=len(unknown),
                highest_unknown_residency=unknown[:32], rank_pin_observations=pins, issues=issues)


def compare_saifs(first, second):
    a, b = first['signals'], second['signals']
    require(a.keys() == b.keys(), 'Signal inventory differs between windows')
    ta, tb = first['duration_ticks'], second['duration_ticks']
    rows = []
    total_difference = total_activity = 0
    for name in sorted(a):
        delta = b[name]['TC'] - a[name]['TC']
        total_difference += abs(delta)
        total_activity += a[name]['TC'] + b[name]['TC']
        p1 = Decimal(b[name]['T1']) / tb - Decimal(a[name]['T1']) / ta
        xz = Decimal(b[name]['TX'] + b[name]['TZ']) / tb - Decimal(a[name]['TX'] + a[name]['TZ']) / ta
        if delta or p1 or xz:
            rows.append(dict(signal=name, transitions_before=a[name]['TC'], transitions_after=b[name]['TC'],
                             transition_delta=delta, high_residency_fraction_delta=str(p1),
                             xz_fraction_delta=str(xz)))
    rows.sort(key=lambda r: (-abs(r['transition_delta']), r['signal']))
    return dict(changed_signals=len(rows), transitions_before=sum(v['TC'] for v in a.values()),
                transitions_after=sum(v['TC'] for v in b.values()),
                summed_absolute_per_signal_transition_difference=total_difference,
                difference_over_combined_transitions=str(Decimal(total_difference) / total_activity),
                largest_transition_changes=rows[:32], statistical_representativeness_established=False)


def analyze_capture(capture):
    capture = Path(capture)
    require((capture / 'CAPTURE_COMPLETE.txt').read_text().splitlines()[0] ==
            'NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED', 'No exact capture marker')
    windows = review_windows(capture / 'windows.tsv')
    parsed = [parse_saif(capture / f'prbs_{i}.saif') for i in range(2)]
    saifs = [summarize_saif(p) for p in parsed]
    comparison = compare_saifs(*parsed)
    return dict(**windows, saifs=saifs, comparison=comparison,
                issues=[f'prbs_{i}: {issue}' for i, summary in enumerate(saifs) for issue in summary['issues']],
                refresh_coverage='NOT_ESTABLISHED_NO_TIME_CORRELATED_COMMAND_TRACE',
                rank_command_coverage='NOT_ESTABLISHED_PIN_OCCUPANCY_ONLY',
                source_files={p.name: digest(p) for p in [capture / 'CAPTURE_COMPLETE.txt',
                             capture / 'windows.tsv', capture / 'prbs_0.saif', capture / 'prbs_1.saif']})


def review(run):
    run = Path(run)
    invocation = json.loads((run / 'invocation.json').read_text())
    require('finished_utc' in invocation, 'Do not review an active run')
    # Keep the sealed review authoritative and retain its diagnostics separately.
    parent = json.loads((run / 'REVIEW.json').read_text())
    require(parent.get('run') == run.name, 'Parent review run identity mismatch')
    result = dict(run=run.name, parent_review_sha256=digest(run / 'REVIEW.json'),
                  invocation_sha256=digest(run / 'invocation.json'),
                  reviewer_sha256=digest(Path(__file__)),
                  parent_status=parent.get('status'), parent_errors=parent.get('errors'),
                  errors=[], pending=['Inspect vendor diagnostics and compiled source identities',
                      'Resolve critical signal omissions and X/Z/TB causes',
                      'Establish rank/refresh coverage with time-correlated command evidence',
                      'Review representativeness; two windows do not prove statistical stability',
                      'Map actual SAIF into the matching DCP and review unmatched nets'],
                  ready_for_power_estimate=False, board_power_qualified=False, exact_dimm_qualified=False)
    if invocation.get('exit_code') != 0 or invocation.get('frozen_source_unchanged') is not True:
        result['errors'].append('Native invocation failed or source changed')
    if invocation.get('source_sha256') != CAPTURE_SOURCE_SHA256:
        result['errors'].append('Invocation does not identify the sealed R1 capture package')
    if parent.get('status') != 'CAPTURE_EXPORTED_REQUIRES_ACTIVITY_REVIEW' or parent.get('errors') != []:
        result['errors'].append('Original capture reviewer did not accept export')
    if parent.get('invocation_sha256') != result['invocation_sha256'] or parent.get('native_identity_observed') is not True:
        result['errors'].append('Parent review invocation binding or native identity is absent/incorrect')
    try:
        captures = [p for p in (run / 'output').rglob('activity_capture') if p.is_dir()]
        require(len(captures) == 1, 'Missing/ambiguous capture directory')
        result['activity'] = analyze_capture(captures[0])
        result['errors'].extend(result['activity']['issues'])
    except (ValueError, KeyError, OSError, ArithmeticError) as exc:
        result['errors'].append('Supplemental review: ' + repr(exc))
    result['status'] = 'ACTIVITY_DETAIL_EXPORTED_REQUIRES_REVIEW' if not result['errors'] else 'DIAGNOSTIC_DDR_ACTIVITY_DETAIL_R1'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Terminal restored run, including original REVIEW.json')
    parser.add_argument('--output', required=True, type=Path, help='New supplemental JSON outside the archived run')
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.run.resolve()), 'Write supplemental review outside archived run')
    result = review(args.run)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], errors=result['errors'], output=str(args.output))))
    return 0 if not result['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
