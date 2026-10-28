"""Add a public AXI response-delay fixture to a PRIVATE native example copy."""
import argparse
import hashlib
import json
import re
from pathlib import Path

NAMES = ('board.v', 'xilinx_dma_pcie_ep.sv', 'sample_tests.vh')


def transform(original):
    board, ep, tests = original
    assert board.count('module board;') == 1 and 'ntrpu_stop_probe' not in board
    board = board.replace('module board;', 'module board;\n    ntrpu_xdma_stop_inventory ntrpu_stop_probe();')
    assert ep.count('  // XDMA target application') == 1
    wiring = '''
    wire ntrpu_app_dma_bvalid, ntrpu_app_dma_bready;
    wire ntrpu_app_bypass_bvalid, ntrpu_app_bypass_bready;
    assign m_axi_bvalid = ntrpu_app_dma_bvalid && !board.ntrpu_stop_probe.hold_dma_b;
    assign ntrpu_app_dma_bready = m_axi_bready && !board.ntrpu_stop_probe.hold_dma_b;
    assign m_axib_bvalid = ntrpu_app_bypass_bvalid && !board.ntrpu_stop_probe.hold_bypass_b;
    assign ntrpu_app_bypass_bready = m_axib_bready && !board.ntrpu_stop_probe.hold_bypass_b;
'''
    ep = ep.replace('  // XDMA target application', wiring+'\n  // XDMA target application')
    for port, signal in [('s_axi_bvalid','ntrpu_app_dma_bvalid'), ('s_axi_bready','ntrpu_app_dma_bready'),
                         ('s_axib_bvalid','ntrpu_app_bypass_bvalid'), ('s_axib_bready','ntrpu_app_bypass_bready')]:
        ep, count = re.subn(r'\.'+port+r'\s*\(m_axi[b]?_b(?:valid|ready)\)', '.'+port+' ('+signal+')', ep)
        assert count == 1, port
    tests = tests[:tests.index('else if(testname')] + '''// Native enumeration/BAR discovery precede this branch.
else if(testname == "ntrpu_stop_inventory") begin
    board.ntrpu_stop_probe.run_probe;
    $finish;
end
'''
    return [board, ep, tests]


def patch(example):
    paths = [example/'imports'/n for n in NAMES]
    original = [p.read_bytes() for p in paths]
    changed = transform([x.decode() for x in original])
    manifest = dict(scope='Single H2C descriptor response inventory and independent BAR write; no reset or chain cancellation qualification.', files=[])
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
