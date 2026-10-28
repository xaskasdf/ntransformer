"""Apply unchanged timing/coverage criteria to a refinement of a reviewed route."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once


def review(out):
    text=Path(__file__).with_name('check_ntrpu_peer_regions_physical.py').read_text(encoding='utf-8')
    marker="    namespace = dict(__name__='regions_physical_base', __file__=__file__)"
    extra='    text = once(text, "need(prior[\'status\'] == \'REGIONS_SYNTHESIS_REVIEWED\' and", "need(prior[\'status\'] == \'REGIONS_ROUTED_INTERNAL_TIMING_OPEN\' and")\n'
    text=once(text,marker,extra+marker)
    ns=dict(__name__='mapped_refine_auditor',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    r=json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert r['refinement']=='post_route_explore' and r['stage']=='route'
    checked=ns['review'](out)
    checked.update(refinement=r['refinement'],input_checkpoint_sha256=r['input_dcp_sha256'])
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
