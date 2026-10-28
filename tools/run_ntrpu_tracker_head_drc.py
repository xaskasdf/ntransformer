"""Audit warnings in the cached-head checkpoint without changing or waiving them."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_tracker_head import verify_profile


if __name__=='__main__':
    verify_profile(Path(__file__).resolve().parents[1])
    text=Path(__file__).with_name('run_ntrpu_peer_shared_drc.py').read_text()
    text=text.replace('peer_shared_drc','tracker_head_drc').replace('SHARED_ROUTED_INTERNAL_TIMING_','TRACKER_HEAD_ROUTED_INTERNAL_TIMING_')
    text=once(text,'sources={n:sha(ROOT/n) for n in SOURCES}',
              "sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}")
    text=once(text,'sources_after={n:sha(ROOT/n) for n in SOURCES}','sources_after={n:sha(ROOT/n) for n in sources}')
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
