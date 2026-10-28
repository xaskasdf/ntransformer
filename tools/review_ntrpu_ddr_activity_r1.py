"""Independent capture diagnostics. Export never certifies board power."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from decimal import Decimal
from prepare_ntrpu_ddr_activity_project_r1 import digest


def saif_review(path):
    assert 0 < path.stat().st_size <= 512 * 1024 * 1024, 'SAIF size outside review limit'
    text = path.read_text()
    stack = []
    duration = None
    scale = None
    signals = {}
    roots = []
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', text):
        token = match[0]
        if token == '(':
            stack.append(dict(atoms=[], fields={}))
        elif token == ')':
            assert stack, 'SAIF parentheses'
            row = stack.pop()
            atoms, fields = row['atoms'], row['fields']
            assert atoms, 'SAIF empty expression'
            key = atoms[0]
            if key == 'DURATION':
                assert duration is None and len(atoms) == 2
                duration = int(atoms[1])
            elif key == 'TIMESCALE':
                assert scale is None and len(atoms) == 3
                scale = (Decimal(atoms[1]), atoms[2])
            elif key in ('T0', 'T1', 'TX', 'TZ', 'TC', 'IG'):
                assert len(atoms) == 2 and stack and key not in stack[-1]['fields']
                stack[-1]['fields'][key] = int(atoms[1])
            elif 'T0' in fields or 'T1' in fields:
                assert len(atoms) == 1 and {'T0', 'T1', 'TX', 'TC'} <= fields.keys()
                assert all(v >= 0 for v in fields.values())
                hierarchy = [s['atoms'][1].strip('"') for s in stack if s['atoms'][:1] == ['INSTANCE']]
                # SAIF may retain its TB root; no memory model/TB nets allowed.
                if hierarchy[:1] == ['ntrpu_ddr_example_activity_tb']:
                    hierarchy = hierarchy[1:]
                assert hierarchy and hierarchy[0] == 'dut', ('Outside DUT', hierarchy, atoms)
                name = '/'.join(hierarchy + atoms)
                assert name not in signals, 'Duplicate SAIF signal'
                signals[name] = fields
            elif key == 'INSTANCE':
                assert len(atoms) == 2
                roots.append(atoms[1])
    assert not stack and duration and scale and signals, 'Incomplete SAIF'
    units = {'s': Decimal(1), 'ms': Decimal('1e-3'), 'us': Decimal('1e-6'),
             'ns': Decimal('1e-9'), 'ps': Decimal('1e-12'), 'fs': Decimal('1e-15')}
    seconds = duration * scale[0] * units[scale[1]]
    assert seconds == Decimal('0.000020480'), ('Unexpected SAIF duration', str(seconds))
    for name, row in signals.items():
        assert sum(row.get(k, 0) for k in ('T0', 'T1', 'TX', 'TZ')) == duration, ('Residency mismatch', name)
    unknown = sum(r.get('TX', 0) + r.get('TZ', 0) for r in signals.values())
    transitions = sum(r['TC'] for r in signals.values())
    assert transitions > 0, 'No SAIF transitions'
    return dict(sha256=digest(path), bytes=path.stat().st_size, duration_ticks=duration,
                timescale=[str(scale[0]), scale[1]], duration_seconds=str(seconds),
                signals=len(signals), signal_names_sha256=hashlib.sha256('\n'.join(sorted(signals)).encode()).hexdigest(),
                transitions=transitions, active_signals=sum(r['TC'] > 0 for r in signals.values()),
                unknown_signal_ticks=unknown, unknown_fraction=str(Decimal(unknown) / (duration * len(signals))),
                signals_with_unknown=sum(r.get('TX', 0) + r.get('TZ', 0) > 0 for r in signals.values()))


def windows_review(path):
    rows = list(csv.DictReader(path.open(newline=''), delimiter='\t'))
    values = {}
    for row in rows:
        key = (row['window'], row['edge'], row['metric'])
        assert key not in values
        values[key] = row['value']
    assert int(values[('all', 'scope', 'objects_selected')]) > 0
    metrics = ('cycles', 'sample_ps', 'aw_count', 'b_count', 'ar_count', 'rlast_count',
               'w_beats', 'r_beats', 'w_bytes', 'w_stalls', 'r_stalls', 'pending_w', 'pending_r')
    windows = []
    for index in range(2):
        name = 'prbs_%d' % index
        edges = {edge: {k: int(values[(name, edge, k)]) for k in metrics} for edge in ('start', 'end')}
        for edge in edges.values():
            assert all(v >= 0 for v in edge.values())
            assert edge['pending_w'] == edge['aw_count'] - edge['b_count']
            assert edge['pending_r'] == edge['ar_count'] - edge['rlast_count']
        delta = {k: edges['end'][k] - edges['start'][k] for k in metrics}
        assert delta['cycles'] == 4096 and delta['sample_ps'] == 20480000
        assert all(delta[k] >= 0 for k in metrics if not k.startswith('pending_'))
        assert delta['w_beats'] + delta['r_beats'] > 0
        assert delta['w_bytes'] <= 64 * delta['w_beats']
        assert delta['w_stalls'] <= 4096 and delta['r_stalls'] <= 4096
        if windows:
            assert windows[-1]['end'] == edges['start'], 'Noncontiguous counters'
            assert values[('prbs_0', 'end', 'simulator_time')] == values[(name, 'start', 'simulator_time')]
        delta['r_bytes'] = delta['r_beats'] * 64
        windows.append(dict(name=name, **edges, delta=delta))
    assert sum(w['delta']['w_beats'] for w in windows) > 0
    assert sum(w['delta']['r_beats'] for w in windows) > 0
    return windows


def review(run):
    run = Path(run)
    errors, pending, windows, saifs = [], [], [], []
    inv = json.loads((run / 'invocation.json').read_text())
    assert 'finished_utc' in inv, 'Do not review an active batch'
    if inv['exit_code'] != 0 or not inv['frozen_source_unchanged']:
        errors.append('Native failure, deadline, or frozen input modification')
    # Scan complete native logs, not only the final export marker.
    diagnostics = {}
    for path in run.rglob('*'):
        if path.is_file() and path.suffix == '.log' and '.runs' not in path.as_posix() and '.cache' not in path.as_posix():
            lines = path.read_text(errors='replace').splitlines()
            hits = [l for l in lines if re.search(r'(?i)(\bERROR\s*:|\bFATAL\s*:|\$fatal|NTRPU_ACTIVITY_\w*ERROR|\bError:)', l)]
            if hits:
                diagnostics[path.relative_to(run).as_posix()] = hits
    if diagnostics:
        errors.append('Native error diagnostics require inspection')
    paths = list((run / 'output').rglob('activity_capture'))
    simlogs = list((run / 'output').rglob('simulate.log'))
    native_identity = False
    for path in simlogs:
        text = path.read_text(errors='replace')
        native_identity |= bool(re.search(r'NTRPU_ACTIVITY_IDENTITY dut_simulation=\s*FALSE tg_simulation=\s*FALSE', text))
    if not native_identity:
        errors.append('Missing native DUT/TG hardware-mode identity')
    try:
        assert len(paths) == 1, 'Missing/ambiguous capture directory'
        capture = paths[0]
        assert (capture / 'CAPTURE_COMPLETE.txt').read_text().startswith('NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED')
        windows = windows_review(capture / 'windows.tsv')
        saifs = [saif_review(capture / ('prbs_%d.saif' % i)) for i in range(2)]
        assert saifs[0]['signal_names_sha256'] == saifs[1]['signal_names_sha256']
    except Exception as exc:
        errors.append('Independent capture review: ' + repr(exc))
    pending = ['Review all vendor diagnostics and compiled source identities',
               'Verify refresh and rank coverage; counters do not establish this',
               'Assess X/Z causes, statistical representativeness and window differences',
               'Map actual SAIF scope into a fresh routed DCP session per window',
               'Report unmatched nets, annotation coverage and inherited thermal assumptions']
    result = dict(status='CAPTURE_EXPORTED_REQUIRES_ACTIVITY_REVIEW' if not errors else 'DIAGNOSTIC_DDR_ACTIVITY_R1',
                  run=run.name, errors=errors, pending=pending, native_identity_observed=native_identity,
                  windows=windows, saifs=saifs, native_error_diagnostics=diagnostics,
                  board_power_qualified=False, exact_dimm_qualified=False,
                  ready_for_power_estimate=False, invocation_sha256=digest(run / 'invocation.json'))
    if len(saifs) == 2:
        result['window_comparison'] = dict(transitions_delta=saifs[1]['transitions'] - saifs[0]['transitions'],
                                         active_signals_delta=saifs[1]['active_signals'] - saifs[0]['active_signals'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    result = review(a.run)
    with (a.run / 'REVIEW.json').open('x') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'], errors=result['errors'])))
