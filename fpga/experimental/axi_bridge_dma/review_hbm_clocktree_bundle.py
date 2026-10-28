"""Verify a restored clock-tree evidence bundle, including its open timing result."""
import argparse
import json
from pathlib import Path
from review_hbm_clocktree import review, sha


def verify_files(root, manifest_name):
    manifest = json.loads((root / manifest_name).read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest) | {manifest_name}:
        raise ValueError('File inventory changed: ' + str(root))
    for name, digest in manifest.items():
        if sha(root / name) != digest:
            raise ValueError('Checksum mismatch: ' + name)


def review_bundle(root):
    verify_files(root, 'MANIFEST.json')
    selection = json.loads((root / 'SELECTION.json').read_text())
    reviews = {}
    for role in ('preflight', 'reopen'):
        reviews[role] = review(root / selection[role])
        if reviews[role] != json.loads((root / (role.upper() + '_REVIEW.json')).read_text()):
            raise ValueError('Review receipt changed: ' + role)
    route = root / selection['route']
    result = json.loads((route / 'RESULT.json').read_text())
    log = (route / 'vivado_console.log').read_text()
    if result['timed_out'] or not result['sources_unchanged']:
        raise ValueError('Incomplete or changed route inputs')
    if result['status'] == 'FAILED':
        if result['returncode'] != 1 or 'Noncanonical reset transport/p_cold_rst_n' not in log:
            raise ValueError('Unclassified original route audit failure')
    elif result['status'] != 'HBM_CLOCKTREE_EXPORTED':
        raise ValueError('Unknown route result')
    checkpoint = route / 'output/routed.dcp'
    if sha(checkpoint) != sha(root / selection['reopen'] / 'source/donor.dcp'):
        raise ValueError('Reopened checkpoint is not the selected routed result')
    if sha(route / 'source/donor.dcp') != sha(root / selection['preflight'] / 'source/donor.dcp'):
        raise ValueError('Preflight and route have different donors')
    phases = json.loads((root / 'PHASES.json').read_text())
    for phase in phases:
        restored = root / phase['restored']
        verify_files(restored, 'FILES.json')
        receipt = json.loads((root / phase['backup']).read_text())
        archive = root / phase['archive']
        if sha(archive) != receipt['archive_sha256'] or archive.stat().st_size != receipt['archive_bytes']:
            raise ValueError('Transferred archive changed')
        r = json.loads((restored / 'RESULT.json').read_text())
        for name, digest in r['sources'].items():
            if sha(restored / 'source' / name) != digest:
                raise ValueError('Phase source snapshot changed')
    negative = root / selection['negative']
    receipt = json.loads((negative / 'REVIEW.json').read_text())
    if receipt['count'] != 22 or len(receipt['checks']) != 22:
        raise ValueError('Missing physical negative controls')
    if review(negative / 'fixture') != reviews['reopen']:
        raise ValueError('Negative-control fixture was not restored')
    for case in receipt['checks']:
        evidence = negative / case['name'] / Path(case['path']).name
        if not case['rejected'] or not case['reason'] or sha(evidence) != case['sha256']:
            raise ValueError('Invalid negative-control receipt')
        if sha(evidence) == sha(negative / 'fixture' / case['path']):
            raise ValueError('Negative control did not alter evidence')
    preflight_negative = root / selection['negative_preflight']
    preflight_receipt = json.loads((preflight_negative / 'REVIEW.json').read_text())
    if preflight_receipt['count'] != 11 or not all(c['rejected'] for c in preflight_receipt['checks']):
        raise ValueError('Missing preflight negative controls')
    if review(preflight_negative / 'fixture') != reviews['preflight']:
        raise ValueError('Preflight negative-control fixture was not restored')
    completed = json.loads((root / 'PROCESS_COMPLETION.json').read_text())
    if {j['phase'] for j in completed['jobs']} != {p['name'] for p in phases}:
        raise ValueError('Incomplete process inventory')
    if any(j['remaining_group'] or j['runner_live'] for j in completed['jobs']):
        raise ValueError('Private VM work remains active')
    donor = root / 'provenance/20261011-hbm-clocked.zip'
    if sha(donor) != '75dd6fa9ed9abb306a1573896227ac9261e662dbc2e3d7382b5b6969dc5a6bad':
        raise ValueError('Canonical composition evidence changed')
    physical = reviews['reopen']
    return dict(status='HBM_CLOCKTREE_BUNDLE_REVIEWED', files=len(json.loads((root/'MANIFEST.json').read_text())),
                phases=len(phases), negative_controls=receipt['count']+preflight_receipt['count'], routed_checkpoint_sha256=sha(checkpoint),
                wns_ns=physical['wns_ns'], whs_ns=physical['whs_ns'],
                internal_timing_met=physical['internal_timing_met'], native_hbm_pcie_ip=False,
                board_qualified=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    print(json.dumps(review_bundle(args.root), indent=2))
