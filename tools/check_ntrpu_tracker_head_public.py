"""Recheck the public ledger without inheriting native qualification across RTL changes."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_tracker_head import verify_profile


def review(out):
    repo=out.parent/'repo';profile=verify_profile(repo)
    text=Path(__file__).with_name('check_ntrpu_peer_regions_public.py').read_text()
    text=once(text,"assert all(r['sources'].get(n)==h for n,h in profile['native_sources'].items())",
              "assert all(r['sources'].get(n)==h for n,h in profile['native_sources'].items() "
              "if n not in ('fpga/experimental/ntrpu_hbm_read_tracker.sv',"
              "'fpga/experimental/ntrpu_hbm_read_adapter.sv'))")
    ns=dict(__name__='head_public_baseline',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    checked=ns['review'](out)
    assert checked['counters']['successful_results']==27
    u=json.loads((repo/'HEAD_UNIT_RESULT.json').read_text());ur=json.loads((repo/'HEAD_UNIT_REVIEW.json').read_text())
    sha=ns['sha']
    assert sha(repo/'HEAD_UNIT_RESULT.json')==profile['unit_result_sha256']==ur['result_sha256']
    assert sha(repo/'HEAD_UNIT_REVIEW.json')==profile['unit_review_sha256']
    assert ur['status']=='TRACKER_HEAD_UNIT_REVIEWED' and u['sources']==u['sources_after']
    assert u['sources']['fpga/experimental/ntrpu_hbm_read_head_tracker.sv']==profile['candidate_sha256']
    checked.update(status='TRACKER_HEAD_PUBLIC_REVIEWED',mapped_native_lineage_verified=False,
                   baseline_lineage_with_explicit_tracker_delta=True,native_requalification_required=True,
                   unit_result_sha256=profile['unit_result_sha256'])
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
