"""Keep the same object-by-object DRC classification for the mapped checkpoint."""
import argparse
import json
from pathlib import Path


def review(out):
    text=Path(__file__).with_name('check_ntrpu_peer_shared_drc.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_drc','peer_regions_drc').replace('SHARED_ROUTED_INTERNAL_TIMING_','REGIONS_ROUTED_INTERNAL_TIMING_')
    text=text.replace('SHARED_DRC_','REGIONS_DRC_')
    ns=dict(__name__='mapped_drc_auditor',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    r=json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert all(ns['sha'](out.parent/'repo'/n)==h for n,h in r['sources'].items())
    return ns['review'](out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
