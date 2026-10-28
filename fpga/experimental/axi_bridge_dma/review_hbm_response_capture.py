"""Independently review archived native logs, payload topology and manifests."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def table(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def audit_topology(root):
    variants = {}
    resources = {}
    for variant in ('reference', 'candidate'):
        directory = root / 'output' / variant
        rows = table(directory / 'enables.tsv')
        expected = {f'{queue}/slot{slot}_reg[{bit}]' for queue in ('response_cut', 'validated_cut')
                    for slot in (0, 1) for bit in range(265)}
        require(len(rows) == 1060 and {r['cell'] for r in rows} == expected,
                'Incomplete or duplicate payload inventory: '+variant)
        for r in rows:
            require(r['primitive'] == 'FDRE' and r['clock'] == 'hbm_clk' and r['enable_driver'],
                    'Payload primitive/clock/driver differs')
            starts = set(r['startpoints'].split(','))
            require(starts and '' not in starts, 'Empty enable cone')
            if r['cell'].startswith('response_cut/'):
                require(('h_rvalid' in starts) == (variant == 'reference'), 'RVALID cone invariant')
                if variant == 'candidate':
                    require('p_rready' not in starts, 'Backpressure enters candidate enable')
                    require(all(s == 'rst_n' or s.startswith('response_cut/') for s in starts),
                            'External input enters free-slot enable')
        owners = table(directory / 'ownership.tsv')
        require({r['cell'] for r in owners} == {'response_cut/write_slot_reg',
                'response_cut/count_reg[0]', 'response_cut/count_reg[1]'}, 'Ownership inventory')
        require(any('h_rvalid' in r['startpoints'].split(',') for r in owners), 'Ownership lost VALID')
        variants[variant] = {r['cell']: r for r in rows}
        cells = table(directory / 'cells.tsv')
        require(len({r['cell'] for r in cells}) == len(cells), 'Duplicate cell inventory')
        resources[variant] = dict(collections.Counter(r['primitive'] for r in cells))
    for cell in variants['reference']:
        if cell.startswith('validated_cut/'):
            for field in ('primitive', 'clock', 'startpoints'):
                require(variants['reference'][cell][field] == variants['candidate'][cell][field],
                        'Validated queue topology changed: '+field)
    return dict(response_registers=530, validated_registers=530,
                reference_enables_from_rvalid=530, candidate_enables_from_rvalid=0,
                validated_enable_cones_unchanged=True, resources=resources)


def review(root):
    manifest = json.loads((root / 'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest) | {'FILES.json'}, 'Archive inventory mismatch')
    for name, digest in manifest.items():
        require(sha(root / name) == digest, 'File digest mismatch: '+name)
    result = json.loads((root / 'RESULT.json').read_text())
    require(result['status'] == 'HBM_RESPONSE_CAPTURE_EXPORTED' and result['error'] is None,
            'Native batch did not complete')
    require(result['sources_unchanged'], 'Sources changed')
    prepared = json.loads((root / 'source/PREPARED.json').read_text())
    require(prepared['response_only'] and not prepared['validated_queue_changed'], 'Wrong scope')
    for name, entry in prepared['files'].items():
        require(sha(root / 'source' / name) == entry['sha256'], 'Prepared source mismatch')
    expected_steps = ['compile', 'queue-elaborate', 'queue-run', 'tracker-elaborate', 'tracker-run']
    for name in ('ownership', 'occupied', 'full'):
        expected_steps += [name+'-compile', name+'-elaborate', name+'-run']
    expected_steps += ['synthesis']
    require([s['name'] for s in result['steps']] == expected_steps, 'Step inventory')
    bad_steps = {'ownership-run', 'occupied-run', 'full-run'}
    for step in result['steps']:
        log = (root / (step['name']+'.log')).read_text(errors='replace')
        require(not step['timed_out'], 'Timed-out evidence')
        if step['name'] in bad_steps:
            require('Fatal:' in log and 'CAPTURE_QUEUE width=' in log, 'Negative control not rejected')
        else:
            require(step['returncode'] == 0 and not re.search(r'(?im)^\s*(error|fatal):', log),
                    'Failed positive step: '+step['name'])
    queue = (root / 'queue-run.log').read_text()
    qr = re.findall(r'CAPTURE_QUEUE PASS width=(\d+) checks=(\d+) pushes=(\d+) pops=(\d+) discarded=(\d+) simultaneous=(\d+) speculative=(\d+) bins=12', queue)
    require(sorted(int(r[0]) for r in qr) == [265, 519] and 'CAPTURE_QUEUE ALL PASS' in queue,
            'Queue simulation coverage')
    for _, checks, pushes, pops, discarded, both, spec in qr:
        require(int(checks) > 100000 and int(pushes) == int(pops)+int(discarded)
                and min(int(discarded), int(both), int(spec)) > 100, 'Queue activity/conservation')
    tracker = (root / 'tracker-run.log').read_text()
    tr = re.findall(r'CAPTURE_TRACKER PASS pending=(\d+) checks=(\d+) delivered=(\d+) descriptors=(\d+) quarantines=(\d+) stalls=(\d+) cold_epochs=(\d+) cycles=(\d+)', tracker)
    require(sorted(int(r[0]) for r in tr) == [1, 2, 4, 8] and 'CAPTURE_TRACKER ALL PASS' in tracker,
            'Tracker simulation coverage')
    for pending, checks, delivered, descriptors, quarantine, stalls, epochs, cycles in tr:
        require(int(checks) > 1000 and int(delivered) == 184*int(pending)+4
                and int(descriptors) == 32*int(pending)+4 and int(quarantine) == 3
                and int(stalls) > 0 and int(epochs) == 6, 'Tracker scenario/conservation coverage')
    controls = json.loads((root / 'NEGATIVE_CONTROLS.json').read_text())
    require([c['name'] for c in controls] == ['ownership', 'occupied', 'full'], 'Negative inventory')
    for control in controls:
        require(control['reason'] in (root / (control['name']+'-run.log')).read_text(), 'Wrong negative failure')
        require(control['sha256'] == sha(root / ('mutation-'+control['name']) /
                'ntrpu_hbm_response_capture_queue.sv'), 'Mutation digest')
    topology = audit_topology(root)
    require((root / 'output/STATUS.txt').read_text().strip() == 'HBM_RESPONSE_CAPTURE_TOPOLOGY_EXPORTED',
            'Missing native topology completion')
    return dict(status='HBM_RESPONSE_CAPTURE_REVIEWED', files=len(manifest),
                queue_cases=[list(map(int, r)) for r in qr], tracker_cases=[list(map(int, r)) for r in tr],
                negative_controls=3, topology=topology,
                checkpoints={v: sha(root / 'output' / v / 'tracker.dcp') for v in ('reference', 'candidate')},
                full_board_integrated=False, physical_timing_qualified=False, native_pcie_hbm_traffic=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = review(a.root.resolve())
    text = json.dumps(result, indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:
            f.write(text)
    print(text)
