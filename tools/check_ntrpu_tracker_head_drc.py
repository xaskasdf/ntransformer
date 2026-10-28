"""Classify every DRC object on the exact routed cached-head checkpoint."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_tracker_head import verify_profile


def review(out):
    verify_profile(out.parent/'repo')
    text=Path(__file__).with_name('check_ntrpu_peer_shared_drc.py').read_text()
    text=text.replace('peer_shared_drc','tracker_head_drc').replace('SHARED_ROUTED_INTERNAL_TIMING_','TRACKER_HEAD_ROUTED_INTERNAL_TIMING_')
    text=text.replace('SHARED_DRC_','TRACKER_HEAD_DRC_')
    ns=dict(__name__='head_drc_auditor',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    r=json.loads((out/'result.json').read_text())
    assert all(ns['sha'](out.parent/'repo'/n)==h for n,h in r['sources'].items())
    return ns['review'](out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
