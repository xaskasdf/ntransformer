"""Recover the serial top from the sealed PIPE loader bank, with a two-way check."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Changed anchor: ' + old[:80])
    return text.replace(old, new, 1)


def prepare(serial, simulated, pipe, out):
    if out.exists():
        raise ValueError('Fresh output required')
    original = serial / 'ntrpu_axi_bridge_hbm_controlled.sv'
    assert sha(original) == '7d6fb94faee14a40f1ae3516ac42b0b87bada975dcef80534e7ec0335a34a1da'
    binding = json.loads((simulated / 'BINDING.json').read_text())
    assert binding['synthesizable_weight_loader'] and not binding['external_preload_ports']
    top = (simulated / 'xilinx_axi_pcie4_ep.v').read_text()
    assert sha(simulated / 'xilinx_axi_pcie4_ep.v') == binding['modified_top_sha256']
    assert sha(simulated / 'ntrpu_hbm_weight_loader.sv') == binding['loader_sha256'] == '14aca9b0b01fbb575671e4399a76a35613010b7a2e16b1e0637b9dbb3bda1103'
    for name, digest in binding['frozen_rtl'].items():
        assert sha(serial / name) == sha(simulated / name) == digest, name
    stock = (pipe / 'xilinx_axi_pcie4_ep.v').read_text()
    assert sha(pipe / 'xilinx_axi_pcie4_ep.v') == binding['pipe_endpoint_sha256']
    blocks = re.findall(r'\s*// synthesis translate_off.*?// synthesis translate_on[^\n]*', stock, re.S)
    assert len(blocks) == 2
    scaffold = re.search(r"  wire  \[25:0\]  common_commands_in_i;.*?  reg pipe_mmcm_rst_n = 1'b1;\n", stock, re.S)[0]
    connections = re.findall(r'^\s*\.(?:common_commands_(?:in|out)|pipe_(?:rx|tx)_\d+_sigs)\s*\([^\n]*\),\s*\n', stock, re.M)
    assert len(connections) == 34
    serial_top = once(top, 'module xilinx_axi_pcie4_ep ', 'module ntrpu_axi_bridge_hbm_controlled ')
    serial_top = once(serial_top, blocks[0] + '\n  input                                       sys_clk_p,', '  input                                       sys_clk_p,')
    serial_top = once(serial_top, scaffold + '\n  wire sys_rst_n_c;', '  wire sys_rst_n_c;')
    serial_top = once(serial_top, ''.join(connections) + '   .sys_clk         ( sys_clk )', '   .sys_clk         ( sys_clk )')
    # Independently compose the physical donor using ONLY the simulated loader
    # attachment and the sixteen native HBM write connections.
    start = serial_top.index('    wire [67:0] load_awaddr;')
    end = serial_top.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_bar32_adapter', start)
    candidate = original.read_text()
    a = candidate.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_control_plane control(')
    b = candidate.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_bar32_adapter', a)
    control = serial_top[serial_top.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_control_plane control('):end]
    assert control == candidate[a:b].replace('(csr_', '(ctl_')
    candidate = candidate[:a] + serial_top[start:end] + candidate[b:]
    a = candidate.index('    ntrpu_hbm_native_rw hbm_binding(')
    pre, body = candidate[:a], candidate[a:]
    for name in ('awaddr', 'awburst', 'awid', 'awlen', 'awsize', 'awvalid', 'awready', 'wdata', 'wstrb', 'wlast', 'wvalid', 'wready', 'bid', 'bresp', 'bvalid', 'bready'):
        body, count = re.subn(r'\.s_' + name + r'\([^)]*\)', f'.s_{name}(load_{name})', body, count=1)
        assert count == 1, name
    candidate = (pre + body).replace('// Read-only native bring-up. This top has no weight-loading path yet.', '// Boot-time BAR loader owns native writes; SEAL enables read commands.')
    assert candidate == serial_top, 'Functional difference beyond simulated loader attachment'
    assert 'pipe_rx_' not in serial_top and 'common_commands_in_i' not in serial_top
    out.mkdir()
    for path in serial.iterdir():
        if path.is_file():
            shutil.copy2(path, out / path.name)
    (out / original.name).write_text(serial_top)
    shutil.copy2(simulated / 'ntrpu_hbm_weight_loader.sv', out / 'ntrpu_hbm_weight_loader.sv')
    script = out / 'elaborate_hbm_native_join.tcl'
    script.write_text(once(script.read_text(), 'ntrpu_hbm_telemetry_pack ntrpu_axi_bridge_hbm_controlled}', 'ntrpu_hbm_telemetry_pack ntrpu_hbm_weight_loader ntrpu_axi_bridge_hbm_controlled}'))
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    report = dict(status='SERIAL_LOADER_COMPOSITION_MATCHES_NATIVE_SIMULATION',
        serial_donor_sha256=sha(original), simulated_top_sha256=binding['modified_top_sha256'],
        physical_top_sha256=sha(out / original.name), loader_sha256=binding['loader_sha256'],
        frozen_rtl=binding['frozen_rtl'], functional_attachment_identical=True,
        simulation_scaffolding_removed=True, top_name='ntrpu_axi_bridge_hbm_controlled',
        part='xcvu47p-fsvh2892-2-e', synthesis_completed=False, board_qualified=False)
    (out / 'WEIGHT_PHYSICAL_BINDING.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('serial', 'simulated', 'pipe', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.serial, a.simulated, a.pipe, a.output)))
