"""Create a private native top from the reviewed bridge example and HBM reader."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from prepare_guard_integration import ports


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    if output.exists():
        raise ValueError('New binding directory required')
    original = source / 'xilinx_axi_pcie4_ep.v'
    text = original.read_text()
    client = source / 'ntrpu_axi_bridge_hbm_clocked.sv'
    parsed = ports(client.read_text())
    private = {n for n in parsed if n.startswith('m_')}
    native = {n for n in parsed if n.startswith('h_')}
    references = re.findall(r'board\.probe\.(\w+)', text)
    if len(references) != 35 or set(references) != private or len(native) != 13:
        raise ValueError('Reviewed interface changed')
    if text.count('module xilinx_axi_pcie4_ep') != 1 or text.count('endmodule') != 1:
        raise ValueError('Unexpected example structure')
    text = text.replace('module xilinx_axi_pcie4_ep', 'module ntrpu_axi_bridge_hbm_native_join', 1)
    text = text.replace('board.probe.', '')
    # This private top must not downgrade findings inherited from the example.
    attribute = '(* DowngradeIPIdentifiedWarnings = "yes" *)'
    if text.count(attribute) != 1:
        raise ValueError('Unexpected example warning attribute')
    text = text.replace(attribute, '')
    # The serial IP has no PIPE simulation ports. Remove the complete example
    # scaffolding, including connections outside its translate_off regions.
    text, blocks = re.subn(r'\s*// synthesis translate_off.*?// synthesis translate_on[^\n]*', '', text, flags=re.S)
    if blocks != 2:
        raise ValueError('Unexpected PIPE simulation blocks')
    text, scaffold = re.subn(r'  wire  \[25:0\]  common_commands_in_i;.*?  reg pipe_mmcm_rst_n = 1\'b1;\n', '', text, flags=re.S)
    text, connections = re.subn(r'^\s*\.(?:common_commands_(?:in|out)|pipe_(?:rx|tx)_\d+_sigs)\s*\([^\n]*\),\s*\n', '', text, flags=re.M)
    if scaffold != 1 or connections != 34 or re.search(r'\b(?:common_commands_\w+|pipe_(?:rx|tx)_\d+_sigs\w*)\b', text):
        raise ValueError('PIPE scaffolding did not match the reviewed example')
    internal = {'clk', 'mgmt_clk', 'native_link_up', 'native_axi_reset_n', 'native_ctl_reset_n',
                'perst_n', 'reader_clk', 'mem_clk', 'init_complete_async'} | private | native
    additions = ['input wire mgmt_clk_in', 'input wire fabric_ref_clk_in',
                 'input wire fabric_clock_reset', 'input wire hbm_ref0_clk_in', 'input wire hbm_ref1_clk_in',
                 'output wire fabric_clock_locked', 'output wire [1:0] hbm_init_complete',
                 'output wire [1:0] hbm_cattrip', 'output wire [13:0] hbm_temperature',
                 'output wire [63:0] hbm_read_parity']
    for name, (direction, width) in parsed.items():
        if name not in internal:
            additions.append(f'{direction} wire {width.replace("AXI_ADDR_BITS", "64")} {name}')
    anchor = '  input                                       sys_rst_n\n);'
    if text.count(anchor) != 1:
        raise ValueError('Example header changed')
    text = text.replace(anchor, '  input                                       sys_rst_n,\n  ' + ',\n  '.join(additions) + '\n);')
    body = []
    for name in sorted(private | native):
        body.append(f'    wire {parsed[name][1].replace("AXI_ADDR_BITS", "64")} {name};')
    body += ['''    wire mgmt_clk,reader_clk,mem_clk,hbm_ref0,hbm_ref1;
    wire [1:0] init_complete_async=hbm_init_complete & {2{fabric_clock_locked}};
    BUFGCE management_buffer(.I(mgmt_clk_in),.CE(1'b1),.O(mgmt_clk));
    BUFGCE hbm_reference0(.I(hbm_ref0_clk_in),.CE(1'b1),.O(hbm_ref0));
    BUFGCE hbm_reference1(.I(hbm_ref1_clk_in),.CE(1'b1),.O(hbm_ref1));
    // Board sequencing must start these clocks and wait for LOCKED before
    // releasing configuration_rst_n. Lock loss is availability, never cold reset.
    fabric_clocks clock_generator(.clk_in1(fabric_ref_clk_in),.reset(fabric_clock_reset),
        .locked(fabric_clock_locked),.clk_out1(reader_clk),.clk_out2(mem_clk));
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_hbm_clocked core(
        .clk(axi_aclk_out),.mgmt_clk(mgmt_clk),
        .native_link_up(user_lnk_up),.native_axi_reset_n(axi_aresetn),
        .native_ctl_reset_n(axi_ctl_aresetn),.perst_n(sys_rst_n_c),.*);
    // Read-only native bring-up. This top has no weight-loading path yet.
    ntrpu_hbm_native_rw hbm_binding(
        .axi_clk(mem_clk),.axi_cold_reset_n(mem_cold_rst_n),
        .s_awaddr(68'b0),.s_awburst(4'b0101),.s_awid(12'b0),.s_awlen(8'b0),
        .s_awsize(6'b101101),.s_awvalid(2'b0),.s_awready(),
        .s_wdata(512'b0),.s_wstrb(64'b0),.s_wlast(2'b0),.s_wvalid(2'b0),.s_wready(),
        .s_bid(),.s_bresp(),.s_bvalid(),.s_bready(2'b11),
        .s_araddr(h_araddr),.s_arburst(h_arburst),.s_arid(h_arid),.s_arlen(h_arlen),
        .s_arsize(h_arsize),.s_arvalid(h_arvalid),.s_arready(h_arready),
        .s_rdata(h_rdata),.s_rdata_parity(hbm_read_parity),.s_rid(h_rid),
        .s_rlast(h_rlast),.s_rresp(h_rresp),.s_rvalid(h_rvalid),.s_rready(h_rready),
        .HBM_REF_CLK_0(hbm_ref0),.HBM_REF_CLK_1(hbm_ref1),
        .APB_0_PWDATA(32'b0),.APB_0_PADDR(22'b0),.APB_0_PCLK(mgmt_clk),
        .APB_0_PENABLE(1'b0),.APB_0_PRESET_N(management_cold_rst_n),.APB_0_PSEL(1'b0),.APB_0_PWRITE(1'b0),
        .APB_1_PWDATA(32'b0),.APB_1_PADDR(22'b0),.APB_1_PCLK(mgmt_clk),
        .APB_1_PENABLE(1'b0),.APB_1_PRESET_N(management_cold_rst_n),.APB_1_PSEL(1'b0),.APB_1_PWRITE(1'b0),
        .APB_0_PRDATA(),.APB_0_PREADY(),.APB_0_PSLVERR(),
        .APB_1_PRDATA(),.APB_1_PREADY(),.APB_1_PSLVERR(),
        .apb_complete_0(hbm_init_complete[0]),.apb_complete_1(hbm_init_complete[1]),
        .DRAM_0_STAT_CATTRIP(hbm_cattrip[0]),.DRAM_1_STAT_CATTRIP(hbm_cattrip[1]),
        .DRAM_0_STAT_TEMP(hbm_temperature[6:0]),.DRAM_1_STAT_TEMP(hbm_temperature[13:7]));
''']
    text = text.replace('endmodule', '\n'.join(body) + '\nendmodule')
    if 'board.probe' in text:
        raise ValueError('Hierarchical testbench dependency survived')
    output.mkdir(parents=True)
    top = output / 'ntrpu_axi_bridge_hbm_native_join.sv'; top.write_text(text)
    pairs = []
    for peer, own in re.findall(r'\.(\w+)\s*\(\s*((?:m_|h_)\w+)\s*\)', text):
        if own not in private | native:
            continue
        width = parsed[own][1]
        if not width:
            size = 1
        elif 'AXI_ADDR_BITS' in width:
            size = 64
        else:
            high, low = map(int, re.fullmatch(r'\[(\d+):(\d+)\]', width).groups())
            size = high - low + 1
        cell = 'ntrpu_bridge_i' if own in private else 'hbm_binding'
        pairs.append((own, f'{cell}/{peer}', size))
    if len(pairs) != 48 or len({p[0] for p in pairs}) != 48:
        raise ValueError('Native wire audit coverage changed')
    (output / 'wire_pairs.tsv').write_text('core_pin\tpeer_pin\twidth\n' +
        ''.join(f'{own}\t{peer}\t{size}\n' for own, peer, size in sorted(pairs)))
    result = dict(status='NATIVE_JOIN_BINDING_PREPARED', example_sha256=sha(original),
        core_sha256=sha(client), memory_binding_sha256=sha(source / 'ntrpu_hbm_native_rw.sv'),
        top_sha256=sha(top), serial_pipe_connections_removed=connections,
        requester_control_ports=sorted(private), hbm_read_ports=sorted(native),
        cold_reset_source='configuration_rst_n only; external sequencer must wait for fabric_clock_locked',
        physical_pinout_qualified=False, native_traffic_qualified=False, board_qualified=False)
    (output / 'BINDING.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); print(json.dumps(prepare(args.source, args.output), indent=2))
