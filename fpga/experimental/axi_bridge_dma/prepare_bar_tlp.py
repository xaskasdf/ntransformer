"""Replace only the incoming BRAM target in a private native PIPE example."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(example, frozen, out):
    if out.exists():raise ValueError('Fresh source directory required')
    out.mkdir(parents=True)
    ep=example/'imports/xilinx_axi_pcie4_ep.v'
    text=ep.read_text()
    text,n=re.subn(r'    //example design BRAM Controller\s+xilinx_axi_pcie_ep_app .*?\n\);','',text,count=1,flags=re.S)
    assert n==1
    names=['awaddr','araddr','awprot','arprot','awvalid','wvalid','bready','arvalid','rready',
           'wdata','wstrb','awready','wready','arready','bvalid','rvalid','bresp','rresp','rdata']
    widths=dict(awaddr=12,araddr=12,awprot=3,arprot=3,wdata=32,wstrb=4,bresp=2,rresp=2,rdata=32)
    body='\n    wire [3:0] bar_bid,bar_rid;\n'
    body+=''.join('    wire '+(f'[{widths[p]-1}:0] ' if p in widths else '')+'csr_'+p+';\n' for p in names)
    body+='    ntrpu_hbm_control_plane control(.clk(axi_aclk_out),.cold_rst_n(board.probe.cold_rst_n),.bus_rst_n(axi_aresetn),\n'
    body+=''.join(f'        .s_{p}(csr_{p}),\n' for p in names)
    body+='''        .cfg_ready(1'b0),.cmd_ready(1'b0),.ack_ready(1'b0),
        .cfg_rsp_valid(1'b0),.cfg_rsp_error(1'b0),.reject_valid(1'b0),.result_valid(1'b0),
        .reject_code(4'b0),.reject_tag(128'b0),.result_tag(128'b0),.result_bytes(13'b0),
        .telemetry_clk(board.probe.telemetry_clk),.telemetry_cold_rst_n({4{board.probe.cold_rst_n}}),
        .telemetry_data(board.probe.telemetry_data));
    ntrpu_bar32_adapter bar_control(.clk(axi_aclk_out),.rst_n(axi_aresetn&&board.probe.cold_rst_n),
'''
    signals={p:'m_axi_'+p for p in ['awaddr','araddr','awlen','arlen','awsize','arsize','awprot','arprot',
        'awburst','arburst','awlock','arlock','awvalid','wvalid','wlast','bready','arvalid','rready',
        'wdata','wstrb','awready','wready','arready','bvalid','rvalid','rlast']}
    signals.update(awid='m_axib_awid',arid='m_axib_arid',bid='bar_bid',rid='bar_rid',bresp='bresp',rresp='rresp',rdata='rdata')
    body+=''.join(f'        .s_{p}({v}),\n' for p,v in signals.items())
    body+=',\n'.join(f'        .m_{p}(csr_{p})' for p in names)+');\n'
    for name in ('bid','rid'):
        text,n=re.subn(r'\.m_axib_'+name+r"\s*\([34]'b0\)",'.m_axib_'+name+'(bar_'+name+')',text)
        assert n==1
    assert text.count('endmodule')==1
    text=text.replace('endmodule',body+'\nendmodule')
    (out/ep.name).write_text(text)
    board=(example/'imports/board.v').read_text()
    assert board.count('module board;')==1 and 'probe();' not in board
    (out/'board.v').write_text(board.replace('module board;','module board;\n    ntrpu_bar_tlp_probe probe();'))
    (out/'board_common.vh').write_bytes((example/'imports/board_common.vh').read_bytes())
    for name in ('ntrpu_bar32_adapter.sv','ntrpu_hbm_control_plane.sv','ntrpu_control_snapshot.sv'):
        (out/name).write_bytes((frozen/name).read_bytes())
    actual=Path('D:/AI/artifacts/ntrpu-axi-bridge-hbm-bar/bar-synth-20261012T002033Z/restored/project/native_join.srcs/sources_1/ip/ntrpu_bridge/ntrpu_bridge.xci')
    params=lambda p: {k:v[0]['value'] for k,v in json.loads(p.read_text())['ip_inst']['parameters']['component_parameters'].items()}
    a=params(example/'ntrpu_bridge_ex.srcs/sources_1/ip/ntrpu_bridge/ntrpu_bridge.xci');b=params(actual)
    differences={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
    assert differences=={'pipe_sim':['true','false']},differences
    for name in ('ntrpu_bar_tlp_probe.sv','run_bar_tlp.py','run_bar_tlp_trace.tcl','prepare_bar_tlp.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'BINDING.json').write_text(json.dumps(dict(example_endpoint_sha256=sha(ep),
        modified_endpoint_sha256=sha(out/ep.name),profile_differences=differences,
        frozen_rtl={name:sha(out/name) for name in ('ntrpu_bar32_adapter.sv','ntrpu_hbm_control_plane.sv','ntrpu_control_snapshot.sv')},
        downstream_producers='Held offers; readiness and responses tied low',
        hbm_memory_simulated=False,physical_link_simulated=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('example','frozen','output'):p.add_argument(name,type=Path)
    a=p.parse_args();prepare(a.example,a.frozen,a.output)
