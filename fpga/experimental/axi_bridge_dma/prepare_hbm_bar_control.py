"""Bind the sealed native top to the CSR and the 1-DW BAR adapter, in a fresh copy."""
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
        raise ValueError('Unexpected anchor: ' + old[:90])
    return text.replace(old, new, 1)


def prepare(donor, boundary, output, helpers):
    if output.exists():
        raise ValueError('Fresh directory required')
    top = donor / 'ntrpu_axi_bridge_hbm_native_join.sv'
    if sha(top) != '8bae8385fcb5f3abd09e390517f0bee49a1c07c643e02f29a59d6971d0178aca':
        raise ValueError('Wrong frozen native top')
    report = json.loads(boundary.read_text())
    assert report['checkpoint_sha256'] == 'd1f97e808bf08cf7e8a1e8298621f9def9ab46836c388c0817caa0cc1f3bdf9f'
    params = json.loads((donor / 'ntrpu_bridge.xci').read_text())['ip_inst']['parameters']['component_parameters']
    selected = {name: params[name][0]['value'] for name in ['functional_mode', 'axi_addr_width', 'axi_data_width',
        'pciebar2axibar_0', 'pf0_bar0_size', 'pf0_bar0_scale', 'pf0_bar0_enabled', 'pf0_bar0_prefetchable']}
    assert selected == dict(functional_mode='AXI_Bridge', axi_addr_width='64', axi_data_width='512_bit',
        pciebar2axibar_0='0x0000000000000000', pf0_bar0_size='128', pf0_bar0_scale='Kilobytes',
        pf0_bar0_enabled='true', pf0_bar0_prefetchable='false')
    text = top.read_text()
    internal = []; ports = []
    for item in report['inventory']:
        if item['role'] not in ('internal_transaction', 'internal_observation'):
            continue
        name = item['signal']
        pattern = r'^  (?:input|output) wire\s+(?:\[[^]]+\]\s*)?' + re.escape(name) + r'[,]?\s*$'
        # Some original declaration lines contain a list of names.
        found = re.search(pattern, text, re.M)
        if found:
            text = text[:found.start()] + text[found.end():]
        else:
            lines = text.splitlines(keepends=True)
            candidates = [i for i, line in enumerate(lines) if re.search(r'^  (?:input|output) wire', line) and
                          re.search(r'\b' + re.escape(name) + r'\b', line)]
            assert len(candidates) == 1, name
            i = candidates[0]
            prefix, names = re.match(r'(  (?:input|output) wire\s+(?:\[[^]]+\]\s*)?)(.*)', lines[i]).groups()
            remaining = [part.strip() for part in names.rstrip(',\r\n').split(',') if part.strip() != name]
            lines[i] = prefix + ','.join(remaining) + ',\n' if remaining else ''
            text = ''.join(lines)
        internal.append('    wire ' + (f"[{item['bits']-1}:0] " if item['bits'] > 1 else '') + name + ';\n')
        ports.append(name)
    text = re.sub(r',\s*\);', '\n);', text, count=1)
    text = once(text, 'module ntrpu_axi_bridge_hbm_native_join ', 'module ntrpu_axi_bridge_hbm_controlled ')
    text = once(text, 'localparam PL_LINK_CAP_MAX_LINK_WIDTH', ''.join(internal) + '\nlocalparam PL_LINK_CAP_MAX_LINK_WIDTH')
    text = once(text, '  ntrpu_bridge ntrpu_bridge_i (', '  wire [3:0] bar_bid,bar_rid;\n  ntrpu_bridge ntrpu_bridge_i (')
    text = once(text, '.m_axib_bid(4\'b0)', '.m_axib_bid(bar_bid)')
    text = once(text, '.m_axib_rid(4\'b0)', '.m_axib_rid(bar_rid)')
    text, removed = re.subn(r'    //example design BRAM Controller\s+xilinx_axi_pcie_ep_app .*?\n\);', '', text, count=1, flags=re.S)
    assert removed == 1
    domains = {'axi_aclk_out': 0, 'management_refclk': 1, 'clk_out1_fabric_clocks': 2, 'clk_out2_fabric_clocks': 3}
    telemetry = sorted((item for item in report['inventory'] if item['role'] == 'internal_observation'), key=lambda item: item['signal'])
    offsets = [32] * 4; assignments = []; mapping = []
    for item in telemetry:
        assert len(item['peer_clocks']) == 1
        domain = domains[item['peer_clocks'][0]]; width = item['bits']; bit = offsets[domain]
        assert bit + width <= 128
        assignments.append(f"    assign data[{domain*128+bit} +: {width}]={item['signal']};")
        mapping.append(dict(signal=item['signal'], domain=domain, bit=bit, width=width, clock=item['peer_clocks'][0]))
        offsets[domain] += width
    pack = '`timescale 1ns/1ps\n`default_nettype none\nmodule ntrpu_hbm_telemetry_pack(\n'
    pack += ''.join('    input wire ' + (f"[{item['bits']-1}:0] " if item['bits'] > 1 else '') + item['signal'] + ',\n' for item in telemetry)
    pack += '    output wire [511:0] data\n);\n' + '\n'.join(assignments) + '\n'
    for domain, offset in enumerate(offsets):
        pack += f"    assign data[{domain*128} +: 32]=32'h4e54010{domain};\n"
        pack += f"    assign data[{domain*128+offset} +: {128-offset}]=0;\n"
    pack += 'endmodule\n`default_nettype wire\n'
    csr_ports = ['awaddr','araddr','awprot','arprot','awvalid','wvalid','bready','arvalid','rready',
                 'wdata','wstrb','awready','wready','arready','bvalid','rvalid','bresp','rresp','rdata']
    widths = dict(awaddr=12,araddr=12,awprot=3,arprot=3,wdata=32,wstrb=4,bresp=2,rresp=2,rdata=32)
    bind = '\n    // Internal control; native inbound address is checked before truncation.\n'
    bind += ''.join('    wire ' + (f'[{widths[p]-1}:0] ' if p in widths else '') + 'csr_' + p + ';\n' for p in csr_ports)
    bind += '    wire [511:0] telemetry_data;\n'
    bind += '    wire [3:0] telemetry_clk={mem_clk,reader_clk,mgmt_clk,axi_aclk_out};\n'
    bind += '    wire [3:0] telemetry_cold_rst_n={mem_cold_rst_n,reader_cold_rst_n,management_cold_rst_n,client_cold_rst_n};\n'
    bind += '    ntrpu_hbm_telemetry_pack telemetry_pack(.data(telemetry_data),.*);\n'
    bind += '    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_control_plane control(\n'
    bind += '        .clk(axi_aclk_out),.cold_rst_n(client_cold_rst_n),.bus_rst_n(axi_aresetn),\n'
    bind += ''.join(f'        .s_{p}(csr_{p}),\n' for p in csr_ports) + '        .*);\n'
    signals = {p: 'm_axi_' + p for p in ['awaddr','araddr','awlen','arlen','awsize','arsize','awprot','arprot',
               'awburst','arburst','awlock','arlock','awvalid','wvalid','wlast','bready','arvalid','rready',
               'wdata','wstrb','awready','wready','arready','bvalid','rvalid','rlast']}
    signals.update(awid='m_axib_awid', arid='m_axib_arid', bid='bar_bid', rid='bar_rid', bresp='bresp', rresp='rresp', rdata='rdata')
    bind += '    (* KEEP_HIERARCHY="YES" *) ntrpu_bar32_adapter bar_control(\n'
    bind += '        .clk(axi_aclk_out),.rst_n(client_cold_rst_n&&axi_aresetn),\n'
    bind += ''.join(f'        .s_{p}({net}),\n' for p, net in signals.items())
    bind += ',\n'.join(f'        .m_{p}(csr_{p})' for p in csr_ports) + ');\n'
    text = once(text, '    wire mgmt_clk,reader_clk,mem_clk,hbm_ref0,hbm_ref1;',
                '    wire mgmt_clk,reader_clk,mem_clk,hbm_ref0,hbm_ref1;\n' + bind)
    output.mkdir(parents=True)
    for path in donor.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    for name in ['ntrpu_control_snapshot.sv','ntrpu_hbm_control_plane.sv','ntrpu_bar32_adapter.sv',
                 'prepare_hbm_bar_control.py','audit_hbm_native_constraints.tcl','constrain_hbm_native_constraints.tcl',
                 'audit_hbm_native_boundary.tcl']:
        shutil.copy2(helpers / name, output / name)
    (output / 'ntrpu_axi_bridge_hbm_controlled.sv').write_text(text)
    (output / 'ntrpu_hbm_telemetry_pack.sv').write_text(pack)
    tcl = (donor / 'elaborate_hbm_native_join.tcl').read_text()
    tcl = tcl.replace('ntrpu_axi_bridge_hbm_native_join', 'ntrpu_axi_bridge_hbm_controlled')
    tcl = once(tcl, 'ntrpu_hbm_native_availability ntrpu_axi_bridge_hbm_controlled}',
               'ntrpu_hbm_native_availability ntrpu_control_snapshot ntrpu_hbm_control_plane ntrpu_bar32_adapter ntrpu_hbm_telemetry_pack ntrpu_axi_bridge_hbm_controlled}')
    tcl = once(tcl, '    read_verilog $source/xilinx_axi_pcie_ep_app.v\n', '')
    (output / 'elaborate_hbm_native_join.tcl').write_text(tcl)
    result = dict(donor_top_sha256=sha(top), top_sha256=sha(output / 'ntrpu_axi_bridge_hbm_controlled.sv'),
        telemetry=mapping, removed_ports=sorted(ports), removed_bits=sum(item['bits'] for item in report['inventory']
            if item['role'] in ('internal_transaction','internal_observation')), native_config=selected,
        native_ip_configuration_changed=False, native_tlp_traffic_tested=False)
    (output / 'BAR_BINDING.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor', type=Path); parser.add_argument('boundary', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = prepare(args.donor, args.boundary, args.output, Path(__file__).resolve().parent)
    print(json.dumps({key: value for key, value in result.items() if key not in ('telemetry','removed_ports')}, indent=2))
