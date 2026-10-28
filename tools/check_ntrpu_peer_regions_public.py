"""Check the same public ledger and address model in the RTL preflight."""
import argparse
import json
from pathlib import Path
from run_ntrpu_peer_read import sha
from check_ntrpu_peer_regions_routed import auditor
from prepare_ntrpu_peer_regions_routed import public_bench


def review(out):
    root=out.parent/'repo';r=json.loads((out/'result.json').read_text(encoding='utf-8'))
    assert r['experiment']=='peer_regions_public' and r['full'] and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert all(sha(root/n)==h for n,h in r['sources'].items())
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    profile=json.loads((root/'REGIONS_PHYSICAL_PROFILE.json').read_text())
    assert all(r['sources'].get(n)==h for n,h in profile['native_sources'].items())
    expected=public_bench(root).replace('ntrpu_peer_regions_physical dut(','ntrpu_peer_regions dut(')
    expected=expected.replace('tb_ntrpu_peer_shared','tb_ntrpu_peer_regions_public')
    assert (root/'fpga/tb/tb_ntrpu_peer_regions_public.sv').read_text(encoding='utf-8')==expected
    text=(out/'suite/simulation.log').read_text();api=auditor()
    return dict(status='REGIONS_PUBLIC_PREFLIGHT_REVIEWED',result_sha256=sha(out/'result.json'),
                seconds=r['seconds'],**api['replay'](text),negative_controls=api['controls'](text),
                mapped_native_lineage_verified=True,post_route_simulated=False,
                native_hbm_controller=False,board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
