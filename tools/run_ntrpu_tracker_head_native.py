"""Bounded native requalification; reuse IP cache but elaborate a new observed top."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once


def driver():
    text=Path(__file__).with_name('run_ntrpu_peer_shared_native.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_native','tracker_head_native').replace('SHARED_NATIVE','HEAD_NATIVE')
    text=once(text,"('PEER_NATIVE_SNAPSHOT_REVIEWED',)","('REGIONS_NATIVE_SNAPSHOT_REVIEWED',)")
    text=once(text,"for old in ('tb_ntrpu_peer_native',",
              "for old in ('tb_ntrpu_peer_regions_full','tb_ntrpu_peer_shared_native','tb_ntrpu_peer_native',")
    text=once(text,"('ntrpu_peer_read_pkg','ntrpu_peer_read_lane','ntrpu_peer_read_admission','ntrpu_peer_read_shared')",
        "('ntrpu_peer_read_pkg','ntrpu_peer_region_pkg','ntrpu_peer_region_lookup','ntrpu_peer_region_admission','ntrpu_peer_region_lane','ntrpu_peer_read_domain','ntrpu_peer_regions')")
    text=once(text,"assert changed <= {'tools/run_ntrpu_tracker_head_native.py'},changed","assert not changed,changed")
    text=once(text,"assert changed in (set(),{'tools/run_ntrpu_tracker_head_native.py'}),changed","assert not changed,changed")
    text=once(text,"    profile=json.loads((ROOT/'HEAD_NATIVE_PROFILE.json').read_text())",
        "    from prepare_ntrpu_tracker_head_native import verify\n    profile=verify(ROOT)")
    return text


if __name__=='__main__':
    exec(compile(driver(),__file__,'exec'),dict(__name__='__main__',__file__=__file__))
