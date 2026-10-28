"""Recheck a restored memory timing bundle and its recorded selection."""
import argparse
import json
from pathlib import Path
from review_hbm_memory_timing import review, sha, domains, constraints, DONOR_SHA
from review_hbm_clocktree import review as physical_review


def verify(root):
    manifest = json.loads((root / 'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'FILES.json'}:
        raise ValueError('Bundle file inventory mismatch')
    if any(sha(root / name) != digest for name, digest in manifest.items()):
        raise ValueError('Bundle file checksum mismatch')
    selection = json.loads((root / 'SELECTION.json').read_text())
    baseline = domains(root / 'baseline/timing.rpt')
    provenance = json.loads((root / 'baseline/PROVENANCE.json').read_text())
    if provenance['donor_sha256'] != DONOR_SHA or provenance['domains'] != baseline:
        raise ValueError('Baseline provenance mismatch')
    reviews, failures = {}, []
    for phase in sorted((root / 'phases').iterdir()):
        result = json.loads((phase / 'RESULT.json').read_text())
        if result['status'] == 'FAILED':
            try:
                review(phase)
            except ValueError as exc:
                if str(exc) != 'Incomplete optimization':
                    raise
                failures.append(phase.name)
            else:
                raise ValueError('Failed phase accepted')
            continue
        checked = review(phase)
        if checked['before'] != baseline:
            raise ValueError('Trial used a different timing baseline')
        if constraints(phase / 'refinement/before.xdc') != constraints(root / 'baseline/effective.xdc'):
            raise ValueError('Trial started with different constraints')
        reviews[phase.name] = checked
    if set(reviews) != set(selection['reviews']) or sorted(failures) != selection['failed_phases']:
        raise ValueError('Phase inventory mismatch')
    if reviews != selection['reviews']:
        raise ValueError('Review receipt disagrees with reports')
    selected = selection['selected_phase']
    if selected is None:
        selected_sha = DONOR_SHA
        timing_met = False
    else:
        selected_sha = reviews[selected]['checkpoint_sha256']
        timing_met = reviews[selected]['internal_timing_met']
    if selection['selected_checkpoint_sha256'] != selected_sha or selection['internal_timing_met'] != timing_met:
        raise ValueError('Selection claim mismatch')
    completion = json.loads((root / 'PROCESS_COMPLETION.json').read_text())
    if any(r['runner_active'] or r['active_own_groups'] for r in completion):
        raise ValueError('Own VM work has not finished')
    if {r['phase'] for r in completion} != set(reviews) | set(failures):
        raise ValueError('Process receipt inventory mismatch')
    negatives = json.loads((root / 'negative/REVIEW.json').read_text())
    if negatives['status'] != 'HBM_CLOCKTREE_NEGATIVES_REJECTED' or negatives['count'] != 22:
        raise ValueError('Incomplete physical negative controls')
    if len(negatives['checks']) != 22 or not all(r['rejected'] for r in negatives['checks']):
        raise ValueError('Invalid evidence was accepted')
    for record in negatives['checks']:
        if sha(root / 'negative' / record['name'] / Path(record['path']).name) != record['sha256']:
            raise ValueError('Negative fixture mismatch')
    if physical_review(root / 'negative/fixture') != reviews[selection['negative_phase']]['physical']:
        raise ValueError('Negative controls used a different physical result')
    if sha(root / 'negative/fixture/source/donor.dcp') != reviews[selection['negative_phase']]['checkpoint_sha256']:
        raise ValueError('Negative controls used a different checkpoint')
    return dict(status='HBM_MEMORY_TIMING_BUNDLE_VERIFIED', successful_phases=len(reviews),
                failed_phases=len(failures), selected_phase=selected, checkpoint_sha256=selected_sha,
                internal_timing_met=timing_met, negative_controls=22,
                native_hbm_pcie_ip=False, board_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
