"""Preserve all timing/CDC thresholds while separately qualifying the head cache."""
import argparse
import json
from pathlib import Path
import re
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_tracker_head import verify_profile


def baseline():
    text=Path(__file__).with_name('check_ntrpu_peer_shared_physical.py').read_text()
    text=text.replace('peer_shared_physical','tracker_head_physical')
    text=text.replace('ntrpu_peer_read_shared core','ntrpu_peer_regions core')
    text=text.replace('SHARED_SYNTHESIS_REVIEWED','TRACKER_HEAD_SYNTHESIS_REVIEWED')
    text=text.replace('SHARED_ROUTED_INTERNAL_TIMING_','TRACKER_HEAD_ROUTED_INTERNAL_TIMING_')
    text=once(text,"prior['status'] == 'SHARED_TRACE_REVIEWED' and prior['mode'] == 'full_behavioral'",
              "prior['status'] == 'TRACKER_HEAD_PUBLIC_REVIEWED' and not prior['native_hbm_controller'] "
              "and prior['counters']['successful_results'] == 27")
    ns=dict(__name__='head_physical_base',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    return ns


def review(out):
    profile=verify_profile(out.parent/'repo');api=baseline();checked=api['review'](out)
    r=json.loads((out/'result.json').read_text())
    heads=[s.split('\t') for s in (out/'suite/head_registers.tsv').read_text().splitlines()]
    assert len(heads)==8 and len({n for n,_ in heads})==8
    assert all(re.search(r'/tracker/head_length_reg\[[0-3]\]$',n) and kind in ('FDRE','FDSE') for n,kind in heads)
    if r['stage']=='synth':
        flow=(out/'flow.tcl').read_text()
        assert '{ntrpu_peer_read_pkg ntrpu_peer_region_pkg ntrpu_peer_region_lookup ' \
               'ntrpu_peer_region_admission ntrpu_peer_region_lane ntrpu_peer_read_domain ntrpu_peer_regions}' in flow
        assert '-top ntrpu_tracker_head_physical' in flow
    checked.update(baseline_lineage_with_explicit_tracker_delta=True,
                   native_requalification_required=True,native_hbm_controller=False,
                   candidate_sha256=profile['candidate_sha256'])
    checked['synthesized_head_registers']=len(heads)
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
