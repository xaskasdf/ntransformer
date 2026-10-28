"""Use the previously reviewed public native port patch in a fresh example."""
import argparse
from pathlib import Path
from patch_ntrpu_hbm_xdma_guarded import patch as guarded_patch


def patch(example):
    guarded_patch(example)
    path=example/'imports/board.v'
    text=path.read_text()
    assert text.count('ntrpu_hbm_xdma_guarded_native probe()') == 1
    path.write_text(text.replace('ntrpu_hbm_xdma_guarded_native probe()', 'ntrpu_hbm_source_native probe()'))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('example', type=Path)
    patch(ap.parse_args().example)
