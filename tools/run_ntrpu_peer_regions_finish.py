"""Resume reviewed placement using the existing bounded 1800-second finish flow."""
from pathlib import Path


if __name__=='__main__':
    text=Path(__file__).with_name('run_ntrpu_peer_shared_finish.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_physical','peer_regions_physical')
    text=text.replace('SHARED_SYNTHESIS_REVIEWED','REGIONS_SYNTHESIS_REVIEWED')
    text=text.replace('SHARED_PHYSICAL_EXPORTED','REGIONS_PHYSICAL_EXPORTED')
    exec(compile(text,__file__,'exec'),dict(__name__='__main__',__file__=__file__))
