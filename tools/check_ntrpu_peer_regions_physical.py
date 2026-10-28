"""Audit mapped OOC implementation without inheriting the earlier timing pass."""
import argparse
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once


def baseline():
    text = Path(__file__).with_name('check_ntrpu_peer_shared_physical.py').read_text(encoding='utf-8')
    text = text.replace('peer_shared_physical', 'peer_regions_physical')
    text = text.replace('ntrpu_peer_read_shared core', 'ntrpu_peer_regions core')
    text = text.replace('SHARED_SYNTHESIS_REVIEWED', 'REGIONS_SYNTHESIS_REVIEWED')
    text = text.replace('SHARED_ROUTED_INTERNAL_TIMING_', 'REGIONS_ROUTED_INTERNAL_TIMING_')
    text = once(text, "prior['status'] == 'SHARED_TRACE_REVIEWED' and prior['mode'] == 'full_behavioral'",
                "prior['status'] == 'REGIONS_NATIVE_TRAFFIC_REVIEWED' and prior['native_hbm_controller'] "
                "and prior['ownership']['counters']['successful_retire'] == 16")
    namespace = dict(__name__='regions_physical_base', __file__=__file__)
    exec(compile(text, __file__, 'exec'), namespace)
    return namespace


def review(out):
    api = baseline(); checked = api['review'](out); need, sha = api['need'], api['sha']
    repo = out.parent / 'repo'
    profile = json.loads((repo / 'REGIONS_PHYSICAL_PROFILE.json').read_text(encoding='utf-8'))
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    need(all(result['sources'].get(n) == h for n, h in profile['native_sources'].items()),
         'mapped native sources changed')
    need(profile['part'] == 'xcvu47p-fsvh2892-2-e' and
         profile['transport_period_ns'] == 5.0 and profile['memory_period_ns'] == 2.222,
         'mapped physical profile changed')
    if result['stage'] == 'synth':
        need(profile['native_result_sha256'] == result['prior_result_sha256'] and
             profile['native_review_sha256'] == sha(out / 'PRIOR_REVIEW.json'), 'native profile binding')
        flow = (out / 'flow.tcl').read_text(encoding='utf-8')
        need('{ntrpu_peer_read_pkg ntrpu_peer_region_pkg ntrpu_peer_region_lookup '
             'ntrpu_peer_region_admission ntrpu_peer_region_lane ntrpu_peer_read_domain ntrpu_peer_regions}'
             in flow and '-top ntrpu_peer_regions_physical' in flow, 'mapped synthesis source selection')
    checked.update(mapped_native_lineage_verified=True,
                   native_result_sha256=profile['native_result_sha256'])
    return checked


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()), sort_keys=True))
