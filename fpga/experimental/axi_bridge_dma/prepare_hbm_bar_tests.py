"""Build a fresh BAR bank, reusing the sealed real-writer control tests through the adapter."""
import argparse
import json
from pathlib import Path
import re
import shutil


def prepare(control, output, helpers):
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True)
    for name in ['ntrpu_hbm_control_plane.sv','ntrpu_control_snapshot.sv','ntrpu_axi_bridge_writer.sv']:
        shutil.copy2(control/name,output/name)
    for name in ['ntrpu_bar32_adapter.sv','tb_hbm_bar32.sv','run_hbm_bar.py','prepare_hbm_bar_tests.py']:
        shutil.copy2(helpers/name,output/name)
    text=(control/'ntrpu_hbm_control_plane.sv').read_text()
    header=text[text.index('module ntrpu_hbm_control_plane'):text.index('\n);')+3]
    header=header.replace('ntrpu_hbm_control_plane','ntrpu_bar_writer_wrapper').replace('output reg','output wire').replace('=0','')
    csr=['awaddr','araddr','awprot','arprot','awvalid','wvalid','bready','arvalid','rready','wdata','wstrb',
         'awready','wready','arready','bvalid','rvalid','bresp','rresp','rdata']
    widths=dict(awaddr=12,araddr=12,awprot=3,arprot=3,wdata=32,wstrb=4,bresp=2,rresp=2,rdata=32)
    body='\n'+''.join('wire '+(f'[{widths[p]-1}:0] ' if p in widths else '')+'c_'+p+';\n' for p in csr)
    body+='wire [511:0] wide_rdata;\nreg [3:0] read_lane=0;\n'
    body+='always @(posedge clk)if(s_arvalid&&s_arready)read_lane<=s_araddr[5:2];\n'
    body+='assign s_rdata=wide_rdata[read_lane*32+:32];\n'
    body+='ntrpu_hbm_control_plane control(.clk(clk),.cold_rst_n(cold_rst_n),.bus_rst_n(bus_rst_n),\n'
    body+=''.join(f'.s_{p}(c_{p}),\n' for p in csr)+'.*);\n'
    body+='ntrpu_bar32_adapter adapter(.clk(clk),.rst_n(cold_rst_n&&bus_rst_n),\n'
    body+=".s_awaddr({52'd0,s_awaddr}),.s_araddr({52'd0,s_araddr}),.s_awid(4'd3),.s_arid(4'd9),\n"
    body+=".s_awlen(8'd0),.s_arlen(8'd0),.s_awsize(3'd2),.s_arsize(3'd2),.s_awburst(2'd1),.s_arburst(2'd1),\n"
    body+=".s_awlock(1'b0),.s_arlock(1'b0),.s_wlast(1'b1),.s_wdata(512'(s_wdata)<<(s_awaddr[5:2]*32)),\n"
    body+=".s_wstrb(64'(s_wstrb)<<(s_awaddr[5:2]*4)),.s_rdata(wide_rdata),.s_bid(),.s_rid(),.s_rlast(),\n"
    body+=''.join(f'.s_{p}(s_{p}),\n' for p in csr if p not in ['awaddr','araddr','wdata','wstrb','rdata'])
    body+=',\n'.join(f'.m_{p}(c_{p})' for p in csr)+');\nendmodule\n'
    (output/'ntrpu_bar_writer_wrapper.sv').write_text('`timescale 1ns/1ps\n`default_nettype none\n'+header+body+'`default_nettype wire\n')
    bench=(control/'tb_hbm_control_plane.sv').read_text()
    bench=bench.replace('tb_hbm_control_plane','tb_hbm_bar_control_unused').replace('tb_hbm_control_writer','tb_hbm_bar_writer')
    bench=bench.replace('hbm_control_harness','hbm_bar_writer_harness').replace('ntrpu_hbm_control_plane dut','ntrpu_bar_writer_wrapper dut')
    anchor='        tick(1);\n        fork\n            begin tick(order==1?4+seed:0);send_aw(address);end'
    assert bench.count(anchor)==1
    bench=bench.replace(anchor,'        s_awaddr=address;\n'+anchor)
    (output/'tb_hbm_bar_writer.sv').write_text(bench)
    (output/'PROFILE.json').write_text(json.dumps(dict(control_donor=str(control),native_tlp_traffic_tested=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();prepare(args.control,args.output,Path(__file__).resolve().parent)
