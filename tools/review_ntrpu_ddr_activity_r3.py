"""Supplemental native capture review: escaped SAIF identifiers and exact Tcl echoes.

Preserves the original REVIEW.json and never authorizes a power estimate.
"""
import argparse
import json
from pathlib import Path
import re

import review_ntrpu_ddr_activity_detail_r1 as detail

ECHO = re.compile(r'^\s*#+\s+puts stderr "NTRPU_ACTIVITY_(?:INTEGRATION|CAPTURE)_ERROR \$message"\s*$')
ERROR = re.compile(r'(?i)(\bERROR\s*:|\bFATAL\s*:|\$fatal|NTRPU_ACTIVITY_\w*ERROR|\bError:)')


def native_diagnostics(run):
    errors, echoes = {}, {}
    for path in sorted(Path(run).rglob('*.log')):
        if '.runs' in path.as_posix() or '.cache' in path.as_posix():
            continue
        name = path.relative_to(run).as_posix()
        for line in path.read_text(errors='replace').splitlines():
            if not ERROR.search(line):
                continue
            if name in ('console.log', 'vivado.log') and ECHO.fullmatch(line):
                echoes.setdefault(name, []).append(line)
            else:
                errors.setdefault(name, []).append(line)
    return errors, echoes


def review(run):
    run = Path(run)
    invocation = json.loads((run/'invocation.json').read_text())
    detail.require('finished_utc' in invocation, 'Capture still active')
    parent = json.loads((run/'REVIEW.json').read_text())
    errors = []
    if invocation.get('exit_code') != 0 or invocation.get('frozen_source_unchanged') is not True:
        errors.append('Native failure or changed frozen source')
    if invocation.get('source_sha256') != detail.CAPTURE_SOURCE_SHA256:
        errors.append('Unexpected capture source identity')
    if parent.get('run') != run.name or parent.get('invocation_sha256') != detail.digest(run/'invocation.json'):
        errors.append('Original review identity or invocation mismatch')
    diagnostics, echoes = native_diagnostics(run)
    if diagnostics:
        errors.append('Actual native error diagnostics require inspection')
    identity = any(re.search(r'NTRPU_ACTIVITY_IDENTITY dut_simulation=\s*FALSE tg_simulation=\s*FALSE',
                            path.read_text(errors='replace')) for path in (run/'output').rglob('simulate.log'))
    if not identity:
        errors.append('Missing native DUT/TG hardware-mode identity')
    result = dict(run=run.name, parent_status=parent.get('status'), parent_errors=parent.get('errors'),
        parent_review_sha256=detail.digest(run/'REVIEW.json'), invocation_sha256=detail.digest(run/'invocation.json'),
        reviewer_sha256=detail.digest(Path(__file__)), detail_reviewer_sha256=detail.digest(Path(detail.__file__)),
        native_error_diagnostics=diagnostics, recognized_tcl_source_echoes=echoes,
        native_identity_observed=identity, errors=errors,
        ready_for_power_estimate=False, board_power_qualified=False, exact_dimm_qualified=False,
        pending=['Complete compiled source identity review', 'Establish rank/refresh command coverage',
                 'Assess noncritical X/Z/TB and representativeness', 'Map SAIF into the matching DCP and report unmatched nets'])
    try:
        captures = [p for p in (run/'output').rglob('activity_capture') if p.is_dir()]
        detail.require(len(captures) == 1, 'Missing or ambiguous capture directory')
        result['activity'] = detail.analyze_capture(captures[0])
        errors.extend(result['activity']['issues'])
    except (ValueError, KeyError, OSError, ArithmeticError) as exc:
        errors.append('Capture review: '+repr(exc))
    result['status'] = 'CAPTURE_ACTIVITY_REVIEWED_IDENTITY_AND_MAPPING_PENDING' if not errors else 'DIAGNOSTIC_DDR_ACTIVITY_R3'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    detail.require(not args.output.resolve().is_relative_to(args.run.resolve()), 'Output must be outside archived run')
    result = review(args.run)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'], errors=result['errors'], output=str(args.output))))
    return 0 if not result['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
