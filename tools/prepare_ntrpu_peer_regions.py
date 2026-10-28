"""Derive a two-region experiment without replacing any qualified reference RTL."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save
from prepare_ntrpu_peer_commit import once

ROOT=Path(__file__).resolve().parents[1]


def variants(donor):
    guard=(donor/'fpga/experimental/ntrpu_peer_read_admission.sv').read_text(encoding='utf-8')
    guard=once(guard,'module ntrpu_peer_read_admission #(','module ntrpu_peer_region_admission #(')
    guard=once(guard,'    output reg cfg_error,','''    input ntrpu_peer_region_pkg::region_map_t cfg_map,
    output wire cfg_map_bad,
    output wire [127:0] admit_address,
    output reg cfg_error,''')
    guard=once(guard,'    lease_t lease;','''    import ntrpu_peer_region_pkg::*;
    lease_t lease;
    region_map_t pinned_map;
    wire [1:0] map_hit;
    wire [63:0] resolved[0:1];
    reg [63:0] saved_address[0:3];
    ntrpu_peer_region_lookup config_map(.lease_cfg(cfg),.region_map(cfg_map),
        .logical_offset(64'b0),.bytes(64'b0),.cfg_bad(cfg_map_bad),.hit(),.physical_address());
''')
    decl='    request_t req[0:1],retiring[0:1],saved[0:3];'
    guard=once(guard,decl,decl+'''
    for(genvar q=0;q<2;q++)begin: mapping
        ntrpu_peer_region_lookup lookup(.lease_cfg(lease),.region_map(pinned_map),
            .logical_offset(req[q].source_offset),.bytes(req[q].bytes),.cfg_bad(),
            .hit(map_hit[q]),.physical_address(resolved[q]));
    end''')
    guard=once(guard,'wire cfg_bad=cfg.source.session==0', 'wire cfg_bad=cfg_map_bad || cfg.source.session==0')
    guard=once(guard,'        assign admit_request[l*1024+:1024]=saved[2*l+issue_ptr[l]];',
               '        assign admit_request[l*1024+:1024]=saved[2*l+issue_ptr[l]];\n        assign admit_address[l*64+:64]=saved_address[2*l+issue_ptr[l]];')
    guard=once(guard,'            for(integer i=0;i<4;i++)if(admission_error[l]==OK',
               '            if(admission_error[l]==OK && !map_hit[l])admission_error[l]=MAP_MISS;\n            for(integer i=0;i<4;i++)if(admission_error[l]==OK')
    guard=once(guard,'            window_end<=0;published_end<=0;',
               "            pinned_map<='0;for(integer i=0;i<4;i++)saved_address[i]<=0;\n            window_end<=0;published_end<=0;")
    guard=once(guard,'                    lease<=cfg;configured<=1;revoked<=revoke;',
               '                    pinned_map<=cfg_map;\n                    lease<=cfg;configured<=1;revoked<=revoke;')
    guard=once(guard,'                        compact[2*l+alloc_ptr[l]]<=',
               '                        saved_address[2*l+alloc_ptr[l]]<=resolved[l];\n                        compact[2*l+alloc_ptr[l]]<=')
    lane=(donor/'fpga/experimental/ntrpu_peer_read_lane.sv').read_text(encoding='utf-8')
    lane=once(lane,'module ntrpu_peer_read_lane #(','module ntrpu_peer_region_lane #(')
    lane=once(lane,'    input ntrpu_peer_read_pkg::request_t request,',
              '    input ntrpu_peer_read_pkg::request_t request,\n    input wire [63:0] request_source_address, // Trusted, captured by region admission.')
    lane=once(lane,"wire [64:0] start_address={1'b0,lease.source_base}+{1'b0,request.source_offset};",
              "wire [64:0] start_address={1'b0,request_source_address};")
    top=(donor/'fpga/experimental/ntrpu_peer_read_shared.sv').read_text(encoding='utf-8')
    top=once(top,'module ntrpu_peer_read_shared (','module ntrpu_peer_regions (')
    top=once(top,'    input wire [1155:0] cfg,','    input wire [1155:0] cfg,\n    input ntrpu_peer_region_pkg::region_map_t cfg_map,')
    top=once(top,'    wire guard_cfg_ready,guard_cfg_error;',
             '    wire guard_cfg_ready,guard_cfg_error,cfg_map_bad;\n    wire [127:0] lane_source_address;')
    top=once(top,'ntrpu_peer_read_admission guard(', 'ntrpu_peer_region_admission guard(\n        .cfg_map(cfg_map),.cfg_map_bad(cfg_map_bad),.admit_address(lane_source_address),')
    top=once(top,'ntrpu_peer_read_lane #(', 'ntrpu_peer_region_lane #(')
    top=once(top,'.cfg_valid(cfg_fire),.cfg_ready(lane_cfg_ready[l])',
             '.cfg_valid(cfg_fire && !cfg_map_bad),.cfg_ready(lane_cfg_ready[l])')
    top=once(top,'.request(lane_request[l*1024+:1024]),',
             '.request(lane_request[l*1024+:1024]),.request_source_address(lane_source_address[l*64+:64]),')
    top=once(top,'    wire [127:0] cmd_addr,cmd_tag,data_tag,cpl_tag;',
             '    wire [1:0] reader_cmd_valid,reader_cmd_ready,domain_pending,domain_fault;\n    wire [127:0] cmd_addr,cmd_tag,data_tag,cpl_tag;')
    top=once(top,'wire source_fault=reader_fault || |lane_reject_valid;',
             'wire source_fault=reader_fault || |lane_reject_valid || |domain_fault;')
    top=once(top,'assign cfg_ready=guard_cfg_ready && (&lane_cfg_ready);',
             'assign cfg_ready=guard_cfg_ready && (&lane_cfg_ready) && !(|domain_pending);')
    top=once(top,'    ntrpu_hbm_read_return reader(','''    for(genvar f=0;f<2;f++)begin: domain_fences
        ntrpu_peer_read_domain fence(.clk(clk),.rst_n(rst_n),.s_valid(cmd_valid[f]),.s_ready(cmd_ready[f]),
            .address(cmd_addr[f*64+:64]),.m_valid(reader_cmd_valid[f]),.m_ready(reader_cmd_ready[f]),
            .complete(cpl_valid[f]&&cpl_ready[f]),.pending(domain_pending[f]),.fault(domain_fault[f]));
    end
    ntrpu_hbm_read_return reader(''')
    top=once(top,'.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),',
             '.cmd_valid(reader_cmd_valid),.cmd_ready(reader_cmd_ready),')
    top=once(top,'.fault_in(|lane_reject_valid),.rst_n',
             '.fault_in(|lane_reject_valid || |domain_fault),.rst_n')
    return dict(ntrpu_peer_region_admission=guard,ntrpu_peer_region_lane=lane,ntrpu_peer_regions=top)


def prepare(selected,output):
    review=json.loads((selected/'REVIEW.json').read_text())
    result=json.loads((selected/'output/result.json').read_text())
    assert review['status']=='SHARED_NATIVE_TRAFFIC_REVIEWED'
    assert review['result_sha256']==sha(selected/'output/result.json')
    assert result['sources']==result['sources_after']
    assert all(sha(selected/'repo'/n)==h for n,h in result['sources'].items())
    assert not output.exists() and not output.resolve().is_relative_to(ROOT)
    shutil.copytree(selected/'repo',output)
    names=['fpga/experimental/ntrpu_peer_region_pkg.sv','fpga/experimental/ntrpu_peer_region_lookup.sv','fpga/experimental/ntrpu_peer_read_domain.sv',
           'fpga/tb/tb_ntrpu_peer_regions_unit.sv','tools/prepare_ntrpu_peer_regions.py',
           'tools/run_ntrpu_peer_regions_unit.py','tools/check_ntrpu_peer_regions_unit.py',
           'fpga/tb/tb_ntrpu_peer_read_domain.sv','tools/run_ntrpu_peer_read_domain.py',
           'tools/check_ntrpu_peer_read_domain.py']
    for n in names:
        shutil.copyfile(ROOT/n,output/n)
    for name,text in variants(selected/'repo').items():
        assert (ROOT/'fpga/experimental'/(name+'.sv')).read_text(encoding='utf-8')==text
        (output/'fpga/experimental'/(name+'.sv')).write_text(text,encoding='utf-8',newline='\n')
    save(output/'REGIONS_PROFILE.json',dict(donor_result_sha256=sha(selected/'output/result.json'),
         donor_sources=result['sources'],map_entries=2,reservations=4,
         generation='Map generation equals lease.source.generation; config blocked while any reservation is held',
         physical_timing_qualified=False,native_hbm_qualified=False,pcie_integrated=False,board_qualified=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('selected',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();prepare(a.selected.resolve(),a.output.resolve())
