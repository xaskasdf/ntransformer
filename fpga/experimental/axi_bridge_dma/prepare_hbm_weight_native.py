"""Attach the boot loader to a fresh copy of the closed native PIPE bank."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from prepare_hbm_tlp import prepare, once


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(donor, pipe, out):
    prepare(donor, pipe, out)
    helper = Path(__file__).parent
    top = (out / 'xilinx_axi_pcie4_ep.v').read_text()
    widths = dict(awaddr=68, awburst=4, awid=12, awlen=8, awsize=6,
                  awvalid=2, awready=2, wdata=512, wstrb=64, wlast=2,
                  wvalid=2, wready=2, bid=12, bresp=4, bvalid=2, bready=2)
    # Remove every preload top port; these are now internal DUT wires.
    for name, width in widths.items():
        top, count = re.subn(r'  (?:input|output) wire \[' + str(width-1) + r':0\] load_' + name + r',\n', '', top)
        assert count == 1, name
    declarations = ''.join(f'    wire [{width-1}:0] load_{name};\n' for name, width in widths.items())
    csr = dict(awaddr=12, araddr=12, awprot=3, arprot=3, awvalid=1, wvalid=1,
               bready=1, arvalid=1, rready=1, wdata=32, wstrb=4, awready=1,
               wready=1, arready=1, bvalid=1, rvalid=1, bresp=2, rresp=2, rdata=32)
    declarations += ''.join(f'    wire [{width-1}:0] ctl_{name};\n' for name, width in csr.items())
    declarations += '    wire weights_sealed;\n'
    start = top.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_control_plane control(')
    end = top.index('    (* KEEP_HIERARCHY="YES" *) ntrpu_bar32_adapter', start)
    top = top[:start] + top[start:end].replace('(csr_', '(ctl_') + top[end:]
    instance = '''    (* KEEP_HIERARCHY="YES" *) ntrpu_hbm_weight_loader weight_loader(
        .clk(axi_aclk_out),.cold_rst_n(client_cold_rst_n),.bus_rst_n(axi_aresetn),
        .mem_clk(mem_clk),.mem_cold_rst_n(mem_cold_rst_n),
        .memory_ready_async((&init_complete_async)&&!availability_fault),.sealed(weights_sealed),
'''
    instance += ''.join(f'        .s_{name}(csr_{name}),.m_{name}(ctl_{name}),\n' for name in csr)
    instance += ',\n'.join(f'        .h_{name}(load_{name})' for name in widths) + ');\n'
    top = top[:start] + declarations + instance + top[start:]
    top = top.replace('// Read-only native bring-up. This top has no weight-loading path yet.',
                      '// Boot-time BAR loader owns native writes; SEAL enables read commands.')
    (out / 'xilinx_axi_pcie4_ep.v').write_text(top)
    board = (out / 'board.v').read_text()
    for name in widths:
        board = once(board, f'        .load_{name}(probe.load_{name}),\n', '')
    (out / 'board.v').write_text(board)
    probe = (helper / 'ntrpu_hbm_tlp_probe.sv').read_text()
    a = probe.index('    reg [67:0] load_awaddr=0;')
    b = probe.index('    integer phase=', a)
    probe = probe[:a] + ''.join(f'    wire [{width-1}:0] load_{name}=`EP.load_{name};\n' for name, width in widths.items()) + probe[b:]
    a = probe.index('    task preload(')
    b = probe.index('    task stage(', a)
    probe = probe[:a] + '''    task preload(input integer lane);
        reg [383:0] image;
        reg [255:0] line_data;
        reg [63:0] tag,address;
        for(integer beat=0;beat<length(lane)/32;beat++)begin
            tag=64'hcafe001000000001+lane*256+beat;address=source_address(lane)+beat*32;
            for(integer i=0;i<32;i++)line_data[i*8+:8]=pattern(lane,beat*32+i);
            image={line_data,tag,address};
            stage('h840,12,image);send_write('h80c,1,15,1,0);
            wait(`EP.weight_loader.done);tick(4);
            check_read('h808,'h12);check_read('h820,tag[31:0]);check_read('h824,tag[63:32]);
            check_read('h828,0);check_read('h82c,address[31:0]);check_read('h830,address[63:32]);
            require(load_b_count[lane]==beat+1&&!`EP.weights_sealed,"attributed HBM B before host completion");
            send_write('h810,1,15,1,0);
            $display("WLT LINE lane=%0d beat=%0d tag=%h address=%h",lane,beat,tag,address);
        end
    endtask
''' + probe[b:]
    probe = once(probe, 'input [351:0] data);', 'input [383:0] data);')
    probe = once(probe, '        preload(0);preload(1);', '''        check_read('h800,32'h4e544c44);check_read('h804,1);check_read('h808,'h10);
        preload(0);preload(1);
        send_write('h814,1,15,1,0);check_read('h808,'h14);check_read('h81c,6);
        require(`EP.weights_sealed&&load_b_count[0]==2&&load_b_count[1]==4,"load fence before read commands");
        $display("WLT SEALED lines=6 bytes=192");''')
    probe = probe.replace('HBM_TLP_PASS', 'HBM_WEIGHT_NATIVE_PASS')
    probe = probe.replace('// Host CSR traffic crosses PIPE; HBM preloading uses public AXI3 write ports only.',
                          '// All payload/control enters through native PCIe BAR; preload ports are observations only.')
    (out / 'ntrpu_hbm_tlp_probe.sv').write_text(probe)
    trace = (helper / 'run_hbm_tlp_trace.tcl').read_text()
    trace = once(trace, 'set objects {};', '''foreach signal {sealed busy done failed rejected request acknowledge completion accepted active shadow written memory_result ms ack_sync req_sync} {
    lappend names /board/AXI_PCIE_EP/weight_loader/$signal
}
set objects {};''')
    (out / 'run_hbm_tlp_trace.tcl').write_text(trace)
    for name in ('ntrpu_hbm_weight_loader.sv', 'run_hbm_weight_native.py', 'prepare_hbm_weight_native.py'):
        shutil.copy2(helper / name, out / name)
    binding = json.loads((out / 'BINDING.json').read_text())
    binding['compile_names'].insert(-3, 'ntrpu_hbm_weight_loader.sv')
    binding.update(modified_top_sha256=sha(out / 'xilinx_axi_pcie4_ep.v'),
                   synthesizable_weight_loader=True, external_preload_ports=False,
                   loader_sha256=sha(out / 'ntrpu_hbm_weight_loader.sv'),
                   changes=['Restored AMD PIPE scaffolding', 'BAR loader CSR router and held-bus CDC',
                            'Internal native HBM writes', 'Read command fence until irreversible SEAL'])
    (out / 'BINDING.json').write_text(json.dumps(binding, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('donor', 'pipe', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    build(a.donor, a.pipe, a.output)
