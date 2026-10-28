"""Drive only the public optional reset in a new, reset-enabled XDMA example."""
import argparse
import hashlib
import json
from pathlib import Path


NAMES = ('board.v', 'xilinx_dma_pcie_ep.sv', 'sample_tests.vh')


def transform(original):
    board, ep, tests = original
    assert board.count('module board;') == 1 and 'ntrpu_reset_probe' not in board
    board = board.replace('module board;', 'module board;\n    ntrpu_xdma_reset_assert ntrpu_reset_probe();')
    needle = ".dma_bridge_resetn ( 1'b1 )"
    assert ep.count(needle) == 1
    ep = ep.replace(needle, '.dma_bridge_resetn ( board.ntrpu_reset_probe.reset_n )')
    index = tests.index('else if(testname')
    tests = tests[:index] + '''// Enumeration and BAR discovery precede this branch in the public TX.
else if(testname == "ntrpu_reset_assert") begin
    board.ntrpu_reset_probe.run_probe;
    $finish;
end
'''
    return [board, ep, tests]


def patch(example):
    paths = [example/'imports'/n for n in NAMES]
    original = [p.read_bytes() for p in paths]
    changed = transform([x.decode() for x in original])
    manifest = dict(scope='Optional reset assertion only; no release, DDR, coordinator or physical qualification.', files=[])
    for path, before, after in zip(paths, original, changed):
        assert before.decode() != after
        path.write_bytes(after.encode())
        manifest['files'].append(dict(path=path.relative_to(example).as_posix(),
            before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (example/'NTRPU_RESET_ASSERT_PATCH.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    patch(parser.parse_args().example)
