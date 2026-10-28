"""Restore PIPE scaffolding around the frozen native top; expose test-only HBM loading."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Anchor changed: '+old[:80])
    return text.replace(old,new,1)


def prepare(donor,pipe,out):
    if out.exists():raise ValueError('Fresh preparation required')
    out.mkdir();frozen={}
    for p in donor.glob('*.sv'):
        if p.name in ('ntrpu_axi_bridge_hbm_controlled.sv','ntrpu_axi_bridge_hbm_native_join.sv','ntrpu_hbm_read_return_cut.sv','tb_hbm_native_availability.sv'):continue
        shutil.copy2(p,out/p.name);frozen[p.name]=sha(p)
    original=donor/'ntrpu_axi_bridge_hbm_controlled.sv';text=original.read_text()
    stock=(pipe/'xilinx_axi_pcie4_ep.v').read_text()
    blocks=re.findall(r'\s*// synthesis translate_off.*?// synthesis translate_on[^\n]*',stock,re.S)
    assert len(blocks)==2
    text=once(text,'module ntrpu_axi_bridge_hbm_controlled ','module xilinx_axi_pcie4_ep ')
    text=once(text,'  input                                       sys_clk_p,',blocks[0]+'\n  input                                       sys_clk_p,')
    scaffold=re.search(r'  wire  \[25:0\]  common_commands_in_i;.*?  reg pipe_mmcm_rst_n = 1\'b1;\n',stock,re.S)[0]
    text=once(text,'  wire sys_rst_n_c;',scaffold+'\n  wire sys_rst_n_c;')
    connections=re.findall(r'^\s*\.(?:common_commands_(?:in|out)|pipe_(?:rx|tx)_\d+_sigs)\s*\([^\n]*\),\s*\n',stock,re.M)
    assert len(connections)==34
    text=once(text,'   .sys_clk         ( sys_clk )',''.join(connections)+'   .sys_clk         ( sys_clk )')
    widths=dict(awaddr=68,awburst=4,awid=12,awlen=8,awsize=6,awvalid=2,awready=2,wdata=512,wstrb=64,wlast=2,wvalid=2,wready=2,bid=12,bresp=4,bvalid=2,bready=2)
    outputs={'awready','wready','bid','bresp','bvalid'}
    declarations=''.join(f"  {'output' if n in outputs else 'input'} wire [{w-1}:0] load_{n},\n" for n,w in widths.items())
    text=once(text,'  input wire  configuration_rst_n',declarations+'  input wire  configuration_rst_n')
    begin=text.index('    ntrpu_hbm_native_rw hbm_binding(')
    pre,body=text[:begin],text[begin:]
    for n in widths:
        body,count=re.subn(r'\.s_'+n+r'\([^)]*\)',f'.s_{n}(load_{n})',body,count=1);assert count==1,n
    text=pre+body
    (out/'xilinx_axi_pcie4_ep.v').write_text(text)
    board=(pipe/'board.v').read_text().replace('ntrpu_bar_tlp_probe probe();','ntrpu_hbm_tlp_probe probe();')
    connections=''.join(f'        .load_{n}(probe.load_{n}),\n' for n in widths)
    connections+='''        .mgmt_clk_in(probe.reference_clk),.fabric_ref_clk_in(probe.reference_clk),
        .hbm_ref0_clk_in(probe.reference_clk),.hbm_ref1_clk_in(probe.reference_clk),
        .fabric_clock_reset(probe.fabric_reset),.configuration_rst_n(probe.cold_rst_n),
'''
    board=once(board,'       AXI_PCIE_EP (','       AXI_PCIE_EP (\n'+connections)
    (out/'board.v').write_text(board);shutil.copy2(pipe/'board_common.vh',out/'board_common.vh')
    helper=Path(__file__).parent
    for name in ('ntrpu_hbm_tlp_probe.sv','run_hbm_tlp.py','run_hbm_tlp_trace.tcl','prepare_hbm_tlp.py'):
        shutil.copy2(helper/name,out/name)
    compile_names=[n+'.sv' for n in ('ntrpu_axi_bridge_guard_v3_reset_export','ntrpu_axi_bridge_hbm_client_reset_export',
        'ntrpu_axi_bridge_hbm_clocked','ntrpu_axi_bridge_hbm_mem_watch','ntrpu_axi_bridge_hbm_source',
        'ntrpu_axi_bridge_hbm_cdc','ntrpu_axi_bridge_endpoint_client','ntrpu_axi_bridge_writer','ntrpu_axi_bridge_mapper',
        'ntrpu_xdma_local_master','ntrpu_bridge_async_fifo','ntrpu_hbm_startup','ntrpu_hbm_read_return',
        'ntrpu_hbm_read_frames','ntrpu_hbm_read_adapter','ntrpu_hbm_native_rw','ntrpu_hbm_native_availability',
        'ntrpu_control_snapshot','ntrpu_hbm_control_plane','ntrpu_bar32_adapter','ntrpu_hbm_telemetry_pack','ntrpu_hbm_tlp_probe')]
    compile_names+=['xilinx_axi_pcie4_ep.v','board.v']
    report=dict(donor_top_sha256=sha(original),modified_top_sha256=sha(out/'xilinx_axi_pcie4_ep.v'),
        pipe_endpoint_sha256=sha(pipe/'xilinx_axi_pcie4_ep.v'),frozen_rtl=frozen,compile_names=compile_names,
        changes=['Restored AMD PIPE scaffolding','Renamed top for stock board','Exposed test-only native HBM write ports'],
        force_or_deposit_in_dut=False,synthesizable_weight_loader=False)
    (out/'BINDING.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('donor','pipe','output'):p.add_argument(name,type=Path)
    a=p.parse_args();prepare(a.donor,a.pipe,a.output)
