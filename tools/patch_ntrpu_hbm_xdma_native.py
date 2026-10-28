"""Connect rev3 to both native C2H channels in a fresh AMD example copy."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def patch(example):
    changes = []
    for name in ('board.v', 'sample_tests.vh', 'xilinx_dma_pcie_ep.sv'):
        path = example / 'imports' / name
        before = path.read_bytes()
        text = before.decode()
        if name == 'board.v':
            assert text.count('module board;') == 1
            text = text.replace('module board;', 'module board;\n    ntrpu_hbm_xdma_native probe();')
        elif name == 'sample_tests.vh':
            text = text[:text.index('else if(testname')] + '''
else if(testname == "ntrpu_hbm_xdma") begin
    board.probe.run_probe;
    $finish;
end
'''
        else:
            a = text.index('  // Descriptor Bypass Control Logic')
            z = text.index('  // Core Top Level Wrapper', a)
            text = text[:a] + text[z:]
            a = text.index('  // XDMA target application')
            # Keep the example AXI-Lite target; disconnect only its stream outputs.
            core, app = text[:a], text[a:]
            for channel in range(2):
                bindings = {'ready': f'dsc_ready[{channel}]', 'load': f'dsc_load[{channel}]',
                            'src_addr': f'dsc_src[{channel*64}+:64]', 'dst_addr': f'dsc_dst[{channel*64}+:64]',
                            'len': f'dsc_len[{channel*28}+:28]', 'ctl': f'dsc_ctl[{channel*16}+:16]'}
                for suffix, signal in bindings.items():
                    core, count = re.subn(r'\.c2h_dsc_byp_'+suffix+'_'+str(channel)+r'\s*\([^)]*\)',
                                         '.c2h_dsc_byp_'+suffix+'_'+str(channel)+'(board.probe.'+signal+')', core)
                    assert count == 1, (channel, suffix)
                core, count = re.subn(r'\.c2h_sts_'+str(channel)+r'\s*\([^)]*\)',
                                     '.c2h_sts_'+str(channel)+f'(board.probe.native_status[{channel*8}+:8])', core)
                assert count == 1
                for suffix, signal, width in [('tdata','axis_data',512),('tkeep','axis_keep',64),
                                              ('tvalid','axis_valid',1),('tlast','axis_last',1),('tready','axis_ready',1)]:
                    part = f'[{channel}]' if width == 1 else f'[{channel*width}+:{width}]'
                    core, count = re.subn(r'\.s_axis_c2h_'+suffix+'_'+str(channel)+r'\s*\([^)]*\)',
                                         '.s_axis_c2h_'+suffix+'_'+str(channel)+'(board.probe.'+signal+part+')', core)
                    assert count == 1
            core, count = re.subn(r'\.h2c_dsc_byp_load_0\s*\([^)]*\)', ".h2c_dsc_byp_load_0(1'b0)", core)
            assert count == 1
            core, count = re.subn(r'\.h2c_dsc_byp_ready_0\s*\([^)]*\)', '.h2c_dsc_byp_ready_0()', core)
            assert count == 1
            text = core + app
        path.write_bytes(text.encode())
        changes.append(dict(path='imports/'+name, before=hashlib.sha256(before).hexdigest(),
                            after=hashlib.sha256(path.read_bytes()).hexdigest()))
    (example / 'NTRPU_NATIVE_PATCH.json').write_text(json.dumps(changes, indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('example', type=Path)
    patch(ap.parse_args().example)
