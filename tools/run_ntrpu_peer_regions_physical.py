"""Use the unchanged bounded physical flow with the mapped top and native donor."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once


def driver():
    text = Path(__file__).with_name('run_ntrpu_peer_shared_physical.py').read_text(encoding='utf-8')
    text = text.replace('peer_shared_physical', 'peer_regions_physical')
    text = text.replace('SHARED_PHYSICAL', 'REGIONS_PHYSICAL')
    text = text.replace('SHARED_SYNTHESIS_REVIEWED', 'REGIONS_SYNTHESIS_REVIEWED')
    text = text.replace('ntrpu_peer_read_shared', 'ntrpu_peer_regions')
    text = once(text, "review['status'] == 'SHARED_TRACE_REVIEWED' and review['mode'] == 'full_behavioral'",
                "review['status'] == 'REGIONS_NATIVE_TRAFFIC_REVIEWED' and review['native_hbm_controller']")
    text = once(text, "review['counters']['successful_retire'] == 27",
                "review['ownership']['counters']['successful_retire'] == 16")
    text = once(text, '{ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_admission ntrpu_peer_regions}',
                '{ntrpu_peer_read_pkg ntrpu_peer_region_pkg ntrpu_peer_region_lookup '
                'ntrpu_peer_region_admission ntrpu_peer_region_lane ntrpu_peer_read_domain ntrpu_peer_regions}')
    return text


if __name__ == '__main__':
    exec(compile(driver(), __file__, 'exec'), dict(__name__='__main__', __file__=__file__))
