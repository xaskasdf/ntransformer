"""Private public-BFM revision: explicit native CC stimulus, preserve old logger diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
from patch_ntrpu_xdma_stop_inventory import transform as original_transform

NAMES = ('board.v','xilinx_dma_pcie_ep.sv','sample_tests.vh','usp_pci_exp_usrapp_tx.v')


def transform(original, include):
    result = original_transform(original[:3])
    tx = original[3]
    first = tx.index('    task TSK_TX_COMPLETION_DATA;')
    last = tx.index('    endtask // TSK_TX_COMPLETION_DATA', first)+len('    endtask // TSK_TX_COMPLETION_DATA')
    replacement = '// NTRPU bounded public CC responder; original task retained in immutable seed.\n`include "'+str(include)+'"'
    result.append(tx[:first]+replacement+tx[last:])
    return result


def patch(example):
    include = Path(__file__).resolve().parents[1]/'fpga/tb/ntrpu_xdma_host_completion_r2.vh'
    assert include.is_file()
    paths = [example/'imports'/n for n in NAMES]
    original = [p.read_bytes() for p in paths]
    changed = transform([x.decode() for x in original], include)
    manifest = dict(scope='Native single H2C inventory and BAR independence; explicit public CC host driver replaces generated completion/logging task. No protected edits.',
                    completion_include=str(include),completion_include_sha256=hashlib.sha256(include.read_bytes()).hexdigest(),files=[])
    for path,before,after in zip(paths,original,changed):
        path.write_bytes(after.encode())
        manifest['files'].append(dict(path=path.relative_to(example).as_posix(),before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (example/'NTRPU_STOP_INVENTORY_PATCH.json').open('x') as stream:
        json.dump(manifest,stream,indent=2);stream.write('\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    patch(parser.parse_args().example)
