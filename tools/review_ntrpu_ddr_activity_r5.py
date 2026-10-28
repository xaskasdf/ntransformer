"""Review R2 SAIF plus passive rank-command VCD; never qualify board power."""
import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

import review_ntrpu_ddr_activity_detail_r1 as detail
import review_ntrpu_ddr_activity_identity_r3 as identity
from review_ntrpu_ddr_activity_r3 import native_diagnostics
import review_ntrpu_ddr_rank_commands_r1 as commands


def rank_evidence(capture, windows):
    """Bind the trace inventory and actual half-open SAIF windows before decoding."""
    capture = Path(capture)
    with (capture/'rank_commands.vcd').open(encoding='utf-8') as stream:
        header = stream.read(2 * 1024 * 1024).split('$enddefinitions', 1)[0]
    scopes = re.findall(r'\$scope\s+\w+\s+(\S+)\s+\$end', header)
    detail.require(bool(scopes) and scopes[0] == 'ntrpu_ddr_example_activity_tb',
                   'Unexpected VCD top scope')
    intervals = []
    for row in windows:
        edges = [Decimal(row[edge]['simulator_seconds']) * Decimal(10**12)
                 for edge in ('start', 'end')]
        detail.require(all(v == v.to_integral_value() for v in edges), 'Noninteger picosecond edge')
        intervals.append(dict(name=row['name'], start_ps=int(edges[0]), end_ps=int(edges[1])))
    trace = commands.review(capture/'rank_commands.vcd', intervals)
    detail.require(trace['all_four_rank_interfaces_present'], 'Four complete representative interfaces required')
    detail.require(trace['trace_first_ps'] <= intervals[0]['start_ps'] - 10000 and
                   trace['trace_last_ps'] >= intervals[-1]['end_ps'] + 10000,
                   'VCD lacks 10 ns of actual extent around the SAIF windows')
    with (capture/'rank_pins.tsv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        detail.require(reader.fieldnames == ['rank', 'pin', 'object'], 'Unexpected rank inventory columns')
        rows = list(reader)
    expected = {(rank, pin): '/ntrpu_ddr_example_activity_tb/'+row['scope']+'/'+pin
                for rank, row in trace['ranks'].items() for pin in commands.PINS}
    found = {}
    for row in rows:
        detail.require(None not in row and None not in row.values(), 'Malformed rank inventory row')
        key = row['rank'], row['pin']
        detail.require(key not in found, 'Duplicate rank inventory pin')
        found[key] = row['object']
    detail.require(found == expected and len(found) == 48, 'Inventory and VCD interfaces differ')
    gaps = []
    for rank, row in trace['ranks'].items():
        for window in row['windows']:
            label = rank+'/'+window['name']
            for kind in ('read', 'write', 'refresh'):
                if not window[kind+'_encoding_observed']:
                    gaps.append(label+': no '+kind+' encoding observed')
            if window['clock_problem_events']:
                gaps.append(label+': ambiguous clock events')
            ambiguous = {name: count for name, count in window['counts'].items()
                         if name in ('AMBIGUOUS_EDGE', 'RESET', 'TEST_MODE_OR_UNKNOWN', 'RESERVED')
                         or name.startswith('UNKNOWN')}
            if ambiguous:
                gaps.append(label+': unresolved encodings '+json.dumps(ambiguous, sort_keys=True))
    trace.update(inventory_sha256=detail.digest(capture/'rank_pins.tsv'),
                 inventory_matches_vcd=True, window_observation_gaps=gaps,
                 saif_windows_bound=True, refresh_schedule_qualified=False)
    return trace


def review(run, source_sha):
    run = Path(run)
    result = dict(run=run.name, errors=[], pending=[], source_sha256=source_sha,
                  ready_for_power_estimate=False, board_power_qualified=False,
                  exact_dimm_qualified=False, capture_revision=2)
    errors = result['errors']
    try:
        detail.require(bool(re.fullmatch('[0-9a-f]{64}', source_sha)), 'Invalid expected source hash')
        invocation = json.loads((run/'invocation.json').read_text())
        detail.require('finished_utc' in invocation, 'Capture still active')
        detail.require(invocation.get('capture_revision') == 2, 'Expected R2 invocation')
        detail.require(invocation.get('source_sha256') == source_sha, 'Unexpected source identity')
        detail.require(invocation.get('exit_code') == 0 and invocation.get('frozen_source_unchanged') is True,
                       'Native failure or changed source')
        result['invocation_sha256'] = detail.digest(run/'invocation.json')
        result['source_identity'] = identity.review(run, source_sha=source_sha)
        errors.extend(result['source_identity']['errors'])
        result['pending'].extend(result['source_identity']['pending'])
        diagnostics, echoes = native_diagnostics(run)
        result.update(native_error_diagnostics=diagnostics, recognized_tcl_source_echoes=echoes)
        if diagnostics:
            errors.append('Actual native error diagnostics require inspection')
        captures = [p for p in (run/'output').rglob('activity_capture') if p.is_dir()]
        detail.require(len(captures) == 1, 'Missing or ambiguous capture directory')
        capture = captures[0]
        result['activity'] = detail.analyze_capture(capture)
        errors.extend(result['activity']['issues'])
        result['rank_commands'] = rank_evidence(capture, result['activity']['windows'])
        result['pending'].extend(result['rank_commands']['window_observation_gaps'])
        # Preserve the SAIF-only helper's scope, explicitly superseded by the VCD
        # for pin observations, never for command acceptance or timing compliance.
        result['activity']['rank_command_coverage'] = 'SEE_BOUND_REPRESENTATIVE_DEVICE_VCD'
        result['activity']['refresh_coverage'] = 'SEE_BOUND_VCD_ENCODINGS_TIMING_COMPLIANCE_PENDING'
    except (ValueError, KeyError, OSError, ArithmeticError) as exc:
        errors.append('R2 capture review: '+repr(exc))
    result['pending'].extend(['Classify X/Z against the netlist', 'Establish representative activity',
        'Associate both SAIF windows with the DCP and inspect unmatched nets',
        'Pin encodings do not prove accepted commands, all-device operation or refresh timing compliance'])
    result['status'] = 'DIAGNOSTIC_DDR_ACTIVITY_R2' if errors else 'DDR_ACTIVITY_R2_REVIEWED_WITH_OPEN_ITEMS'
    result['reviewer_sha256'] = detail.digest(Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run, args.source_sha)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], errors=result['errors'])))
    raise SystemExit(bool(result['errors']))
