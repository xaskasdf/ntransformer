"""Run the mapped public netlist regression with the unchanged 900-second bound."""
from pathlib import Path


if __name__ == '__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_shared_routed.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_routed','peer_regions_routed')
    text=text.replace('SHARED_ROUTED_INTERNAL_TIMING_PASS','REGIONS_ROUTED_INTERNAL_TIMING_PASS')
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
