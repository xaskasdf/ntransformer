"""Chain fixture: local management and bounded BAR0 discovery; adjacent H2C descriptors."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from patch_ntrpu_xdma_c2h_stop import NAMES, transform as c2h_transform

SEED_EP_SHA256 = '2f14d850ef5753312707b3467e3f47b5c06666ad1963d52614d90879aef0cc81'
LOCAL_SUFFIXES = ('awaddr awprot awvalid awready wdata wstrb wvalid wready '
                  'bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready').split()


def transform(original, include):
    assert hashlib.sha256(original[1].encode()).hexdigest() == SEED_EP_SHA256
    changed = c2h_transform(original, include)
    ep = changed[1]
    start = ep.index('  ntrpu_xdma ntrpu_xdma_i')
    end = ep.index('    );', start) + len('    );')
    core = ep[start:end]
    for suffix in LOCAL_SUFFIXES:
        core, count = re.subn(r'\.s_axil_' + suffix + r'\s*\(m_axil_' + suffix + r'\)',
                             '.s_axil_' + suffix + ' (board.ntrpu_stop_probe.local_' + suffix + ')', core)
        assert count == 1, suffix
    for channel in ('h2c', 'c2h'):
        core, count = re.subn(r'\.' + channel + r'_sts_0\s*\(\)',
                             '.' + channel + '_sts_0 (board.ntrpu_stop_probe.' + channel + '_status)', core)
        assert count == 1, channel
    # AMD loops m_axil to s_axil instead of instantiating its RAM slave here.
    # BAR0 is used once by AMD discovery; return the original zero RAM value.
    # The bench permits one address-zero read before phase1 and no later request.
    idle = '''
    assign m_axil_awready = 1'b0;
    assign m_axil_wready = 1'b0;
    assign m_axil_bvalid = 1'b0;
    assign m_axil_bresp = 2'b00;
    assign m_axil_arready = board.ntrpu_stop_probe.init_arready;
    assign m_axil_rvalid = board.ntrpu_stop_probe.init_rvalid;
    assign m_axil_rdata = 32'b0;
    assign m_axil_rresp = 2'b00;
'''
    changed[1] = ep[:start] + core + idle + ep[end:]
    return changed


def patch(example):
    include = Path(__file__).resolve().parents[1] / 'fpga/tb/ntrpu_xdma_chain_completion.vh'
    assert include.is_file()
    paths = [example / 'imports' / n for n in NAMES]
    raw = [p.read_bytes() for p in paths]
    changed = transform([x.decode() for x in raw], include)
    manifest = dict(scope='Public local AXI-Lite and channel status; native BAR2/H2C/C2H response holds. One BAR0 discovery read returns zero before phase1; subsequent BAR0 requests prohibited. No reset or protected state edits.',
                    completion_include=str(include), completion_include_sha256=hashlib.sha256(include.read_bytes()).hexdigest(), files=[])
    for path, before, after in zip(paths, raw, changed):
        path.write_bytes(after.encode())
        manifest['files'].append(dict(path=path.relative_to(example).as_posix(),
                                      before_sha256=hashlib.sha256(before).hexdigest(),
                                      after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with (example / 'NTRPU_STOP_INVENTORY_PATCH.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    patch(parser.parse_args().example)
