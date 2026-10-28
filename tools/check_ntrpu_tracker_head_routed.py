"""Audit payload and ownership against the exact timing-passing netlist."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_tracker_head import verify_profile


def review(out):
    verify_profile(out.parent/'repo')
    text=Path(__file__).with_name('check_ntrpu_peer_regions_routed.py').read_text()
    text=text.replace('REGIONS_ROUTED_','TRACKER_HEAD_ROUTED_')
    text=text.replace('ntrpu_peer_regions_physical','ntrpu_tracker_head_physical')
    text=text.replace('from prepare_ntrpu_peer_regions_routed import','from prepare_ntrpu_tracker_head_routed import')
    ns=dict(__name__='head_routed_auditor',__file__=__file__);exec(compile(text,__file__,'exec'),ns)
    assert json.loads((out/'result.json').read_text())['experiment']=='tracker_head_routed'
    checked=ns['review'](out)
    from prepare_ntrpu_tracker_head_routed import PORTS
    meta=json.loads((out/'suite/COMPARISON.json').read_text())
    assert meta['public_connection_map']==PORTS
    assert meta['flattened_map_port_widths']=={p:2 if p=='cfg_map[valid]' else 64 for p in PORTS}
    checked.update(post_route_simulated=True,native_requalification_required=True)
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
