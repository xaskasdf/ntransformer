"""Freeze two simultaneously authorized regions and a domain-fenced HBM bench."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_peer_regions import prepare,ROOT
from run_ntrpu_peer_read import save


def full(selected,output):
    prepare(selected,output)
    for n in ('tools/prepare_ntrpu_peer_regions_full.py','tools/run_ntrpu_peer_regions_full.py',
              'tools/check_ntrpu_peer_regions_full.py','fpga/tb/ntrpu_peer_regions_body.svh'):
        shutil.copyfile(ROOT/n,output/n)
    text=(selected/'repo/fpga/tb/tb_ntrpu_peer_shared_native.sv').read_text(encoding='utf-8')
    text=once(text,'module tb_ntrpu_peer_shared_native;','module tb_ntrpu_peer_regions_full;')
    text=once(text,'    lease_t cfg;','    import ntrpu_peer_region_pkg::*;\n    lease_t cfg;\n    region_map_t cfg_map;')
    text=once(text,'ntrpu_peer_read_shared dut(','ntrpu_peer_regions dut(')
    text=once(text,'task automatic configure;','task automatic configure(input bit expect_error=0);')
    text=once(text,'check(!cfg_error,"CFG_ERROR")','check(cfg_error==expect_error,"CFG_ERROR")')
    text=once(text,'value_at(cfg.source_base+observed.source_offset+',
              'value_at(expected_address(observed.source_offset,observed.source.generation)+')
    start=text.index('    request_t a,b,c,d,v;')
    text=text[:start]+(ROOT/'fpga/tb/ntrpu_peer_regions_body.svh').read_text(encoding='utf-8')+'\nendmodule\n'
    extra='''    // Independent bench map for this fixed two-generation preload.
    function automatic [63:0] expected_address(input logic[63:0] offset,generation);
        if(generation==1)return offset>=32768?64'h200000000+offset-32768:offset;
        return offset>=32768?offset-32768:64'h200000000+offset;
    endfunction
    always @(posedge clk)if(rst_n)begin
        if(cfg_valid&&cfg_ready)$display("RG MAP %0d %0d %h",$rtoi($realtime*1000),epoch,cfg_map);
        for(integer l=0;l<2;l++)begin
            if(dut.lane_request_valid[l]&&dut.lane_request_ready[l])
                $display("RG FWD %0d %0d %0d %h %0d",$rtoi($realtime*1000),epoch,l,
                    dut.lane_request[l*1024+:1024],dut.lane_source_address[l*64+:64]);
            $display("RG DOMAIN %0d %0d %0d %b %b %0d %b %b %b %b %b",
                $rtoi($realtime*1000),epoch,l,dut.cmd_valid[l],dut.reader_cmd_ready[l],dut.cmd_addr[l*64+:64],
                dut.cmd_ready[l],dut.reader_cmd_valid[l],dut.cpl_valid[l]&&dut.cpl_ready[l],
                dut.domain_pending[l],dut.domain_fault[l]);
        end
    end
'''
    text=once(text,'    task automatic check',extra+'    task automatic check')
    (output/'fpga/tb/tb_ntrpu_peer_regions_full.sv').write_text(text,encoding='utf-8',newline='\n')
    tcl=(selected/'repo/tools/ntrpu_peer_shared_native.tcl').read_text(encoding='utf-8')
    tcl=tcl.replace('ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_admission ntrpu_peer_read_shared',
        'ntrpu_peer_read_pkg ntrpu_peer_region_pkg ntrpu_peer_region_lookup ntrpu_peer_region_admission ntrpu_peer_region_lane ntrpu_peer_read_domain ntrpu_peer_regions')
    tcl=tcl.replace('tb_ntrpu_peer_shared_native','tb_ntrpu_peer_regions_full')
    (output/'tools/ntrpu_peer_regions_full.tcl').write_text(tcl,encoding='utf-8',newline='\n')
    tcl=(selected/'repo/tools/ntrpu_peer_shared_native_run.tcl').read_text(encoding='utf-8')
    tcl=tcl.replace('tb_ntrpu_peer_shared_native','tb_ntrpu_peer_regions_full').replace('SHARED_NATIVE','REGIONS_NATIVE')
    (output/'tools/ntrpu_peer_regions_full_run.tcl').write_text(tcl,encoding='utf-8',newline='\n')
    profile=json.loads((output/'SHARED_NATIVE_PROFILE.json').read_text())
    profile.update(scope='Two active mapped regions; per-ingress physical-domain drain; original RTL retained as reference only',
                   region_map_entries=2,integrated_timing_qualified=False,pcie_integrated=False,board_qualified=False)
    save(output/'REGIONS_NATIVE_PROFILE.json',profile)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('selected',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();full(a.selected.resolve(),a.output.resolve())
