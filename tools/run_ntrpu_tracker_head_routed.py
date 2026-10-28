"""Run the mapped public netlist regression within its original 900-second bound."""
from pathlib import Path
from prepare_ntrpu_tracker_head import verify_profile


if __name__=='__main__':
    verify_profile(Path(__file__).resolve().parents[1])
    text=Path(__file__).with_name('run_ntrpu_peer_shared_routed.py').read_text()
    text=text.replace('peer_shared_routed','tracker_head_routed')
    text=text.replace('SHARED_ROUTED_INTERNAL_TIMING_PASS','TRACKER_HEAD_ROUTED_INTERNAL_TIMING_PASS')
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
