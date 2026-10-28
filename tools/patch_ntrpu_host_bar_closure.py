"""Add a public AXI response-delay fixture to a PRIVATE native example copy."""
import argparse
import hashlib
import json
import re
from pathlib import Path

NAMES = ('board.v', 'xilinx_dma_pcie_ep.sv', 'sample_tests.vh')


from patch_ntrpu_xdma_stop_inventory import transform


def patch(example):
    paths = [example/'imports'/n for n in NAMES]
    original = [p.read_bytes() for p in paths]
    changed = transform([x.decode() for x in original])
    manifest = dict(scope='Native BAR host submission closure model with generation ledger, delayed public B and same-address readback. No SGDMA, driver, universal fence or reset qualification.', files=[])
    for path, before, after in zip(paths, original, changed):
        path.write_bytes(after.encode())
        manifest['files'].append(dict(path=path.relative_to(example).as_posix(),
            before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (example/'NTRPU_STOP_INVENTORY_PATCH.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    patch(parser.parse_args().example)
