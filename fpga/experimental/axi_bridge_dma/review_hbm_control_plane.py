"""Review closed control-plane evidence, with negative controls for the reviewer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

GROUPS = [
    ['axil_transport', 'decode_partial', 'descriptor_atomic', 'response_mailboxes',
     'command_ack_retention', 'snapshot_rounds', 'snapshot_stopped_clocks', 'hot_cold_reset'],
    ['mapping_commit', 'writer_transfer', 'writer_rejections', 'writer_ack_fault'],
]
TAG = '12345678abcdef09000000010000000'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed(root, phase):
    files = json.loads((root / 'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(files) | {'FILES.json'}, 'Closed inventory mismatch')
    for name, digest in files.items():
        require(sha(root / name) == digest, 'Hash mismatch: ' + name)
    result = json.loads((root / 'RESULT.json').read_text())
    require(result['status'] == 'CONTROL_' + phase.upper() + '_EXPORTED' and result['error'] is None, 'Run failed')
    require(result['sources_unchanged'], 'Sources changed')
    require(not result['native_bar_attached'] and not result['physical_cdc_qualified'] and
            not result['native_pcie_hbm_traffic'], 'Unsupported integration/physical claim')
    require(all(not step['returncode'] and not step['timed_out'] for step in result['steps']), 'Unsuccessful step')
    require(result['sources'] == {p.name: sha(p) for p in (root / 'source').iterdir() if p.is_file()}, 'Source inventory mismatch')
    profile = json.loads((root / 'source/PROFILE.json').read_text())
    require(profile['writer_sha256'] == sha(root / 'source/ntrpu_axi_bridge_writer.sv'), 'Frozen writer mismatch')
    return result


def parse_trace(text, writer, seed):
    require(not re.search(r'(?im)^\s*(?:error|fatal):|CP CHECK|CP WATCHDOG', text), 'Trace contains failure')
    matches = re.findall(r'^CP PASS writer=(\d+) seed=(\d+) groups=(\d+) checks=(\d+) cfg=(\d+) cmd=(\d+) ack=(\d+) result=(\d+) reject=(\d+)$', text, re.M)
    require(len(matches) == 1, 'Expected exactly one PASS')
    values = tuple(map(int, matches[0]))
    require(values[:3] == (writer, seed, len(GROUPS[writer])) and values[3] >= 500, 'Wrong identity/coverage')
    require(values[4:] == ((1, 4, 2, 1, 2) if writer else (1, 2, 1, 1, 1)), 'Downstream counts changed')
    groups = re.findall(r'^CP GROUP writer=(\d+) seed=(\d+) name=(\w+)$', text, re.M)
    require(groups == [(str(writer), str(seed), name) for name in GROUPS[writer]], 'Groups missing, reordered or duplicated')
    events = re.findall(r'^CP EVENT writer=(\d+) kind=(\d+) tag=([0-9a-f]+) info=(\d+)$', text, re.M)
    expected = [(str(writer), '2', TAG + '1', '128')] * 2 + [(str(writer), '1', TAG + '2', '3')]
    if writer:
        expected.append(('1', '1', TAG + '3', '1'))
    require(events == expected, 'Mailbox identity/bytes/order changed')
    if writer:
        b_only = re.findall(r'^CP B_ONLY tag=([0-9a-f]+) owned=(\d) result=(\d) fault=(\d)$', text, re.M)
        require(b_only == [(TAG + '1', '1', '0', '0'), (TAG + '4', '1', '0', '0')], 'Missing B-only retention')
        bad_ack = re.findall(r'^CP BAD_ACK owned=(\d) fault=(\d) code=(\d+) result=(\d)$', text, re.M)
        require(bad_ack == [('1', '1', '8', '0')], 'Missing bad-ACK ownership retention')
    else:
        stopped = re.findall(r'^CP STOPPED domain=(\d) progress=([0-9a-f]+)$', text, re.M)
        require(stopped == [('1', 'd'), ('2', 'b'), ('3', '7')], 'Incomplete stopped-clock coverage')
    return dict(writer=writer, seed=seed, groups=len(groups), checks=values[3], downstream=list(values[4:]))


def check_structure(structure):
    require(structure['ASYNC_REG'] == 24, 'Expected four pairs of three-stage synchronizers')
    require(structure['LATCH'] == 0, 'Inferred latch')
    require(structure['FDCE'] + structure['FDRE'] > 1000 and structure['LUT'] > 0, 'OOC structure missing')


def review(simulation, synthesis):
    sim = closed(simulation, 'sim'); synth = closed(synthesis, 'synth')
    require(sim['sources'] == synth['sources'], 'Simulation/synthesis source revisions differ')
    require(sim['seeds'] == [0, 1, 2, 3], 'Full phase/stall bank required')
    expected_steps = ['compile']
    traces = {}; records = []
    for writer in range(2):
        expected_steps.append('elaborate-' + str(writer))
        for seed in range(4):
            name = f'writer-{writer}-seed-{seed}'
            expected_steps.append(name)
            text = (simulation / (name + '.log')).read_text(errors='replace')
            traces[writer, seed] = text
            records.append(parse_trace(text, writer, seed))
    require([step['name'] for step in sim['steps']] == expected_steps, 'Incomplete run matrix')
    require(sim['completed'] == [{k: item[k] for k in ('writer', 'seed', 'groups', 'checks')} for item in records], 'Completed summary disagrees with traces')
    require([step['name'] for step in synth['steps']] == ['synth'], 'Wrong synthesis steps')
    structure = {}
    for line in (synthesis / 'output/structure.tsv').read_text().splitlines()[1:]:
        name, value = line.split('\t'); structure[name] = int(value)
    check_structure(structure)
    synth_log = (synthesis / 'synth.log').read_text(errors='replace')
    require(len(re.findall(r'^CONTROL_OOC_SYNTH_COMPLETE$', synth_log, re.M)) == 1, 'Missing synthesis marker')
    require(not re.search(r'(?im)^\s*(?:CRITICAL WARNING|ERROR):', synth_log), 'Unexpected synthesis critical/error')
    cdc_text = (synthesis / 'output/cdc_unconstrained.rpt').read_text()
    cdc = {name: dict(severity=severity, count=int(count)) for name, severity, count in
           re.findall(r'^(CDC-\d+)\s+(\w+)\s+(\d+)\s+', cdc_text, re.M)}
    require(cdc == {'CDC-3': dict(severity='Info', count=8), 'CDC-15': dict(severity='Warning', count=512)},
            'OOC CDC structure changed; do not silently waive new findings')
    timing_text = (synthesis / 'output/timing_unconstrained.rpt').read_text()
    timing_segment = timing_text.split('| Design Timing Summary', 1)[1].split('| Clock Summary', 1)[0]
    timing_numbers = re.findall(r'^\s+(-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+\d+\s+.*)$', timing_segment, re.M)
    require(len(timing_numbers) == 1, 'Timing evidence missing')
    timing_values = timing_numbers[0].split()
    missing_delays = {name: int(re.search(r'checking ' + name + r' \((\d+)\)', timing_text).group(1))
                      for name in ('no_input_delay', 'no_output_delay')}
    require(all(value > 0 for value in missing_delays.values()), 'Expected open OOC boundary constraints')
    negative = []

    def reject(name, callback):
        try:
            callback()
        except (ValueError, KeyError):
            negative.append(name)
        else:
            raise ValueError('Reviewer negative control escaped: ' + name)

    unit = traces[0, 0]; writer = traces[1, 0]
    reject('missing_pass', lambda: parse_trace(unit.replace('CP PASS', 'REMOVED PASS'), 0, 0))
    reject('duplicate_pass', lambda: parse_trace(unit + re.search(r'^CP PASS.*$', unit, re.M).group(0) + '\n', 0, 0))
    reject('wrong_seed', lambda: parse_trace(unit, 0, 1))
    reject('missing_group', lambda: parse_trace(unit.replace('name=descriptor_atomic', 'name=absent'), 0, 0))
    reject('changed_count', lambda: parse_trace(writer.replace('cfg=1 cmd=4', 'cfg=1 cmd=5'), 1, 0))
    reject('changed_event_identity', lambda: parse_trace(writer.replace(TAG + '2', TAG + '9'), 1, 0))
    reject('changed_event_bytes', lambda: parse_trace(writer.replace('info=128', 'info=64'), 1, 0))
    reject('missing_b_retention', lambda: parse_trace(writer.replace('CP B_ONLY', 'REMOVED B_ONLY'), 1, 0))
    reject('bad_ack_released', lambda: parse_trace(writer.replace('CP BAD_ACK owned=1', 'CP BAD_ACK owned=0'), 1, 0))
    reject('missing_stopped_clock', lambda: parse_trace(unit.replace('domain=3 progress=7', 'domain=3 progress=f'), 0, 0))
    reject('failure_after_pass', lambda: parse_trace(unit + '\nFatal: injected\n', 0, 0))
    wrong = copy.deepcopy(structure); wrong['ASYNC_REG'] -= 1
    reject('lost_synchronizer', lambda: check_structure(wrong))
    wrong_latch = copy.deepcopy(structure); wrong_latch['LATCH'] = 1
    reject('inferred_latch', lambda: check_structure(wrong_latch))
    return dict(status='CONTROL_PLANE_REVIEWED', simulations=records,
                total_checks=sum(item['checks'] for item in records), total_groups=sum(item['groups'] for item in records),
                structure=structure, negative_controls=negative,
                unconstrained_cdc=cdc, missing_io_delays=missing_delays,
                estimated_unqualified_timing=dict(wns_ns=float(timing_values[0]), whs_ns=float(timing_values[4]),
                                                  wpws_ns=float(timing_values[8])),
                checkpoint_sha256=sha(synthesis / 'output/control.dcp'), sources=sim['sources'],
                sim_seconds=sim['seconds'], synth_seconds=synth['seconds'],
                native_bar_attached=False, physical_cdc_qualified=False,
                native_pcie_hbm_traffic=False, telemetry_status_packing_integrated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulation', type=Path)
    parser.add_argument('synthesis', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(review(args.simulation, args.synthesis), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
