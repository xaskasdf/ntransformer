"""Use the bounded native driver with a new top, new profile, and old cache provenance."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once


if __name__=='__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_shared_native.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_native','peer_regions_full').replace('SHARED_NATIVE','REGIONS_NATIVE')
    text=once(text,"('PEER_NATIVE_SNAPSHOT_REVIEWED',)","('SHARED_NATIVE_SNAPSHOT_REVIEWED',)")
    text=once(text,"for old in ('tb_ntrpu_peer_native',", "for old in ('tb_ntrpu_peer_shared_native','tb_ntrpu_peer_native',")
    text=once(text,"('ntrpu_peer_read_pkg','ntrpu_peer_read_lane','ntrpu_peer_read_admission','ntrpu_peer_read_shared')",
        "('ntrpu_peer_read_pkg','ntrpu_peer_region_pkg','ntrpu_peer_region_lookup','ntrpu_peer_region_admission','ntrpu_peer_region_lane','ntrpu_peer_read_domain','ntrpu_peer_regions')")
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
