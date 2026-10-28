"""Add public AXI W backpressure and lease tracing to the frozen HBM bench."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from prepare_ntrpu_hbm_native_transport import ROOT, replace, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('reference', type=Path)
    ap.add_argument('--beats', type=int, choices=(8,64), default=8)
    a = ap.parse_args()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_hbm_native_transport.py'),
                    str(a.output), str(a.reference)], check=True)
    source = a.output/'generated/tb_ntrpu_hbm_native_transport.sv'
    text = source.read_text()
    text = replace(text, 'module tb_ntrpu_hbm_native_transport;', 'module tb_ntrpu_hbm_reserve;')
    # Gates start closed, open monotonically during a wave, and close only at
    # complete drain. No force, internal IP access, DUT edits or warm reset.
    text = replace(text, '    ntrpu_hbm_native_rw hbm(', '''    reg [1:0] write_open=0;
    wire [1:0] memory_wready;
    assign h_wready=memory_wready & write_open;
    ntrpu_hbm_native_rw hbm(''')
    text = replace(text, '.s_wvalid(h_wvalid)', '.s_wvalid(h_wvalid & write_open)')
    text = replace(text, '.s_wready(h_wready)', '.s_wready(memory_wready)')
    text = text[:text.index('    initial begin\n        #4700;')]
    text += f'    localparam integer FRAME_BEATS={a.beats};\n'
    text += (ROOT/'fpga/tb/ntrpu_hbm_reserve_body.svh').read_text()+'\nendmodule\n'
    target = a.output/'generated/tb_ntrpu_hbm_reserve.sv'
    target.write_text(text)
    composition = json.loads((a.output/'COMPOSITION.json').read_text())
    composition.update(bench_sha256=sha(target), cases=2, expected_descriptors=48, frame_beats=a.beats,
                       testbench_write_gate=True, production_rtl_modified=False)
    (a.output/'COMPOSITION.json').write_text(json.dumps(composition, indent=2)+'\n')


if __name__ == '__main__':
    main()
