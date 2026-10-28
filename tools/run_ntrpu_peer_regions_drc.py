"""Inspect mapped checkpoint warnings without changing the checkpoint or waivers."""
from pathlib import Path
from prepare_ntrpu_peer_commit import once


if __name__=='__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_shared_drc.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_drc','peer_regions_drc').replace('SHARED_ROUTED_INTERNAL_TIMING_','REGIONS_ROUTED_INTERNAL_TIMING_')
    text=once(text,"sources={n:sha(ROOT/n) for n in SOURCES}",
              "sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}")
    text=once(text,"sources_after={n:sha(ROOT/n) for n in SOURCES}","sources_after={n:sha(ROOT/n) for n in sources}")
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
