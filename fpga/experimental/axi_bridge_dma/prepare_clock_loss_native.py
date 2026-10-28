"""Prepare a private native clock-loss fixture locally; never access the VM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE_SHA='1d9c164a2b0a3a84e46055df79f421f68b658ae1fde333e86202f877fe207664'
GUARD_SHA='d5b340ab788143cdca33e4cec760778722e76c783cb5e0e3ee63798df74d142d'


def fixture(base):
    text=base.decode()
    old='''    reg cold_rst_n=0;
    wire link_up=board.AXI_PCIE_EP.user_lnk_up && board.AXI_PCIE_EP.axi_aresetn && board.AXI_PCIE_EP.axi_ctl_aresetn;
    reg stop=0;'''
    new='''    reg configuration_rst_n=0,mgmt_clk=0;
    always #5000 mgmt_clk=~mgmt_clk;
    wire cold_rst_n,link_up,stop,guard_fault,guard_alive,guard_armed;
    wire [1:0] guard_code;
    wire native_available=board.AXI_PCIE_EP.user_lnk_up && board.AXI_PCIE_EP.axi_aresetn && board.AXI_PCIE_EP.axi_ctl_aresetn;
    ntrpu_axi_bridge_guard guard(.mgmt_clk(mgmt_clk),.axi_clk(clk),.cold_rst_n(configuration_rst_n),
        .native_link_up(board.AXI_PCIE_EP.user_lnk_up),.native_axi_reset_n(board.AXI_PCIE_EP.axi_aresetn),
        .native_ctl_reset_n(board.AXI_PCIE_EP.axi_ctl_aresetn),.perst_n(board.sys_rst_n),
        .client_cold_rst_n(cold_rst_n),.client_link_up(link_up),.client_stop(stop),
        .clock_alive(guard_alive),.armed(guard_armed),.fault(guard_fault),.fault_code(guard_code));'''
    text=text.replace('\r\n','\n')
    if text.count(old)!=1:raise ValueError('Native fixture clock/reset binding changed')
    text=text.replace('module ntrpu_axi_bridge_native;','module ntrpu_axi_bridge_clock_loss;',1).replace(old,new,1)
    old_start='''        require(link_up===1'b1,"enumerated endpoint with both resets released");
        tick(16);cold_rst_n=1;phase=1;tick(4);'''
    new_start='''        require(native_available===1'b1,"enumerated endpoint with both resets released");
        tick(16);configuration_rst_n=1;wait(cold_rst_n && link_up);phase=1;tick(4);'''
    if text.count(old_start)!=1:raise ValueError('Native startup changed')
    text=text.replace(old_start,new_start,1)
    start=text.index('        // Retain a third command across an operational PCIe reset;')
    end=text.index('    endtask',start)
    text=text[:start]+'''        // Own a third command before injecting only the public AXI clock fault.
        phase=3;cmd_tag={64'd9,64'd3};cmd_generation=2;cmd_source=64'h1000;cmd_offset=0;cmd_bytes=64;
        cmd_valid=1;do @(posedge clk);while(!cmd_ready);
        @(negedge clk);cmd_valid=0;wait(src_req_valid);tick(2);
        run_clock_loss;
'''+text[end:]
    text=text.replace('    task run_probe;','    `include "clock_loss_scenario.svh"\n    task run_probe;',1)
    return text


def prepare(output):
    base=(HERE/'ntrpu_axi_bridge_native.sv').read_bytes()
    if hashlib.sha256(base).hexdigest()!=BASE_SHA:raise ValueError('Unreviewed native base')
    if hashlib.sha256((HERE/'ntrpu_axi_bridge_guard.sv').read_bytes()).hexdigest()!=GUARD_SHA:
        raise ValueError('Unreviewed guard')
    output.mkdir(parents=True,exist_ok=False)
    files=['ntrpu_axi_bridge_native.sv','ntrpu_axi_bridge_guard.sv','ntrpu_axi_bridge_writer.sv',
           'ntrpu_axi_bridge_mapper.sv','ntrpu_axi_bridge_endpoint_client.sv','clock_loss_scenario.svh',
           'prepare_clock_loss_native.py','run_clock_loss_native.py','run_clock_loss_trace.tcl',
           'review_clock_loss_native.py','review_native.py','check_clock_loss_reviewer.py','CLOCK_LOSS.md']
    for name in files:shutil.copy2(HERE/name,output/name)
    shutil.copy2(HERE.parents[1]/'rtl/ntrpu_xdma_local_master.sv',output/'ntrpu_xdma_local_master.sv')
    (output/'ntrpu_axi_bridge_clock_loss.sv').write_text(fixture(base),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output/'PREPARED.json').write_text(json.dumps(dict(status='PREPARED_NOT_NATIVE_VALIDATED',
        files=hashes,native_base_sha256=BASE_SHA,guard_sha256=GUARD_SHA),indent=2)+'\n',encoding='utf-8')
    return hashes


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    args=ap.parse_args();print(json.dumps(dict(status='PREPARED_NOT_NATIVE_VALIDATED',files=len(prepare(args.output)))))
