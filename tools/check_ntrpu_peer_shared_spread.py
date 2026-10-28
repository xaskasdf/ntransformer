"""Verify the single placement-flow change and the shared physical acceptance gates."""
import argparse
import json
from pathlib import Path
from check_ntrpu_peer_shared_physical import review as physical_review, sha, need


def review(out):
    result = json.loads((out / 'result.json').read_text())
    repo = out.parent / 'repo'
    profile = json.loads((repo / 'PLACEMENT_PROFILE.json').read_text())
    baseline = repo / 'PLACEMENT_BASELINE.tcl'
    need(sha(baseline) == profile['baseline_flow_sha256'], 'placement baseline hash')
    for name, digest in json.loads((out / 'FILES.json').read_text()).items():
        need(sha(out / name) == digest, 'artifact changed: ' + name)
    need(result['sources'] == result['sources_after'] and result['donor_unchanged'], 'source or donor changed')
    need(all(sha(repo / n) == h for n, h in result['sources'].items()), 'source hash')
    before = baseline.read_text().splitlines()[1:]
    old = '    place_design'
    new = '    place_design -directive AltSpreadLogic_high'
    need(before.count(old) == 1, 'baseline placement command')
    expected = [new if line == old else line for line in before]
    need((out / 'flow.tcl').read_text().splitlines()[1:] == expected, 'unexpected Tcl change')
    need(result['placement_directive'] == 'AltSpreadLogic_high', 'placement directive')
    need(result['input_dcp_sha256'] == profile['synthesis_checkpoint_sha256'], 'selected synthesis')
    prior = json.loads((out / 'PRIOR_REVIEW.json').read_text())
    need(prior['result_sha256'] == result['prior_result_sha256'] == sha(out / 'DONOR_RESULT.json'), 'donor binding')
    need(prior['checkpoint_sha256'] == result['input_dcp_sha256'] == sha(out / 'input.dcp'), 'input checkpoint')
    if result['status'] == 'EXPORTED_PENDING_REVIEW':
        checked = physical_review(out)
    else:
        need(result['status'] == 'FAILED_OR_CHANGED', 'unfinished experiment')
        checked = dict(status='SPREAD_FAILED_BATCH_ARCHIVED', result_sha256=sha(out / 'result.json'),
                       returncode=result['returncode'], timed_out=result['timed_out'],
                       qualified_timing=False)
    checked.update(single_placement_change_verified=True, flow_sha256=sha(out / 'flow.tcl'),
                   placement_directive='AltSpreadLogic_high')
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(review(parser.parse_args().output.resolve()), sort_keys=True))
