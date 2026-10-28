"""Review routed timing with strict provenance and explicit fixture limits."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(folder, placed):
    output = folder/'output'
    record = json.loads((output/'result.json').read_text())
    need(record['status'] == 'EXPORTED_PENDING_REVIEW' and record['returncode'] == 0 and not record['timed_out'],
         'Completed router and report export required')
    need(record['sources'] == record['sources_after'], 'Sources preserved during run')
    for name, sha in record['sources'].items():
        need(digest(folder/'inputs'/name) == sha, 'Source snapshot mismatch: '+name)
    checkpoint = 'candidate_place.dcp'
    sha = record['resume']['checkpoints'][checkpoint]
    need(digest(placed/checkpoint) == sha == digest(output/'inputs'/checkpoint), 'Placement provenance')
    previous = json.loads((placed/'result.json').read_text())
    need(previous['status'] == 'EXPORTED_PENDING_REVIEW' and previous['sources'] == previous['sources_after'],
         'Previously completed unchanged placement')
    logic = [n for n in record['sources'] if n.startswith('fpga/') or 'prepare_ntrpu_' in n]
    need(all(record['sources'][n] == previous['sources'][n] for n in logic), 'Same synthesized logic')
    console = (output/'console.log').read_text()
    need('NTRPU_CANDIDATE_ROUTE_EXPORTED' in console and
         not re.search(r'(?m)^ERROR:', console), 'Clean completed export marker')
    need((output/'route_command.txt').read_text().startswith('RETURN_CODE 0\n'), 'Router command completed')
    fields = dict(line.split(' ', 1) for line in (output/'metrics.txt').read_text().splitlines())
    need(fields['CLOCK_PERIOD'] == '5.000' and fields['PARENT_CLOCK_LOC'] == 'BUFGCE_X0Y60', 'Fixed clock context')
    need(all(fields[k] == v for k, v in {'CLOCKS': '1', 'BUFGCE': '1', 'LATCHES': '0', 'BLACKBOXES': '0'}.items()),
         'Expected physical structure')
    timing = (output/'timing.rpt').read_text()
    checks = {k: int(v) for k, v in re.findall(r'checking (\w+) \((\d+)\)', timing)}
    keys = ('no_clock', 'constant_clock', 'unconstrained_internal_endpoints', 'multiple_clock', 'loops', 'latch_loops')
    need(all(k in checks for k in keys), 'Timing coverage report complete')
    match = re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)', timing, re.M)
    need(match is not None and len(match[1].split()) == 12, 'Complete setup/hold/pulse summary')
    values = list(map(float, match[1].split()))
    routes = {}
    text = (output/'route_status.rpt').read_text()
    for label in ('# of routable nets', '# of fully routed nets', '# of nets with routing errors'):
        found = re.search(re.escape(label)+r'\.+\s*:\s*(\d+)', text)
        need(found is not None, 'Missing route counter '+label)
        routes[label] = int(found[1])
    with (output/'drc_violations.tsv').open(newline='') as stream:
        drc = list(csv.DictReader(stream, delimiter='\t'))
    xdc_before = (output/'constraints_before.xdc').read_text()
    xdc_after = (output/'constraints_after.xdc').read_text()
    timing_commands = lambda t: [line.strip() for line in t.splitlines()
        if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s', line)]
    need(timing_commands(xdc_before) and timing_commands(xdc_before) == timing_commands(xdc_after), 'Timing budgets unchanged')
    forbidden = re.findall(r'(?m)^\s*(set_false_path|set_multicycle_path|set_clock_groups|set_max_delay|set_min_delay|set_disable_timing)\b', xdc_before+'\n'+xdc_after)
    need(not forbidden, 'No timing exceptions or path overrides allowed')
    need(not re.search(r'(?m)^\s*(set_property\s+SEVERITY|create_waiver)\b', xdc_after), 'No DRC severity overrides/waivers')
    internal = {k: float(fields[k]) for k in ('INTERNAL_SETUP', 'INTERNAL_HOLD')}
    gates = dict(
        complete_routing=routes['# of routable nets'] > 0 and routes['# of routable nets'] == routes['# of fully routed nets'] and routes['# of nets with routing errors'] == 0,
        internal_timing=all(v >= 0 for v in internal.values()),
        setup_hold_pulse_summary=all(values[i] >= 0 for i in (0, 4, 8)) and all(values[i] == 0 for i in (2, 6, 10)),
        internal_coverage=all(checks[k] == 0 for k in keys),
        no_drc_errors=not any(row['severity'] in ('Error', 'Critical Warning') for row in drc),
        no_critical_console_warnings=not re.search(r'(?m)^CRITICAL WARNING:', console))
    result = dict(status='ROUTED_INTERNAL_TIMING_PASS_INTEGRATION_PENDING' if all(gates.values()) else 'ROUTED_INTERNAL_NOT_QUALIFIED',
        gates=gates, clock_period_ns=5.0, parent_clock_loc=fields['PARENT_CLOCK_LOC'], internal=internal,
        summary=dict(setup_ns=values[0], hold_ns=values[4], pulse_ns=values[8], setup_failing_endpoints=values[2],
                     hold_failing_endpoints=values[6], pulse_failing_endpoints=values[10]),
        routing=routes, timing_checks=checks, drc=drc, warnings=dict(Counter(re.findall(r'(?m)^WARNING: \[([^]]+)\]', console))),
        seconds=record['seconds'], cpus=record['cpus'], nice=record['nice'], checkpoint_input_sha256=sha,
        routed_checkpoint_sha256=digest(output/'candidate_route_attempt.dcp'),
        netlist_sha256=digest(output/'candidate_core_funcsim.v'),
        limitations=['Registered timing sampler is not a production ready/valid adapter.',
                    'External sampler I/O is unconstrained; no board I/O, PCIe/P2P, HBM or MIG qualification.',
                    'RTL protocol evidence comes from the prior unchanged design; routed netlist simulation is separate.',
                    'Single clock/reset epoch; no CDC or warm-recovery qualification.'])
    for kind in ('setup', 'hold'):
        text = (output/('internal_'+kind+'.rpt')).read_text()
        result[kind+'_path'] = {key: re.search(pattern, text).group(1).strip() for key, pattern in (
            ('source', r'Source:\s*([^\n]+)'), ('destination', r'Destination:\s*([^\n]+)'), ('logic_levels', r'Logic Levels:\s*(\d+)'))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--placed', type=Path, required=True)
    args = ap.parse_args()
    result = review(args.run, args.placed)
    result['checker_sha256'] = digest(Path(__file__))
    with (args.run/'REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
