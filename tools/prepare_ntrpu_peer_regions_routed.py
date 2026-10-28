"""Exercise both mapped stacks using only public ports of an unchanged netlist."""
import hashlib
import json
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_peer_shared_routed import public_bench as shared_bench


def public_bench(root):
    text = shared_bench(root).replace('ntrpu_peer_shared_physical', 'ntrpu_peer_regions_physical')
    text = once(text, '    lease_t cfg;', '    import ntrpu_peer_region_pkg::*;\n    lease_t cfg;\n    region_map_t cfg_map,saved_map;')
    text = once(text, '(address>>16)^', '(address>>16)^(address>>24)^(address>>32)^')
    text = once(text, 'value_at(cfg.source_base+observed.source_offset+',
                'value_at(mapped_address(observed.source_offset,observed.source.generation)+')
    text = once(text, '    task automatic configure;', '''    function automatic [63:0] mapped_address(input logic [63:0] offset,generation);
        if(generation==1)return offset>=16384?64'h200000000+offset-16384:offset;
        return offset>=16384?offset-16384:64'h200000000+offset;
    endfunction
    always @(posedge clk)if(rst_n && cfg_valid && cfg_ready)
        $display("SP MAP %0d %0d %h",$rtoi($realtime*1000),epoch,cfg_map);
    task automatic configure;
        cfg_map='0;cfg_map.generation=cfg.source.generation;cfg_map.valid=3;
        cfg_map.first.logical_base=0;cfg_map.first.bytes=16384;
        cfg_map.second.logical_base=16384;cfg_map.second.bytes=16384;
        cfg_map.first.physical_base=cfg.source.generation==1?0:64'h200000000;
        cfg_map.second.physical_base=cfg.source.generation==1?64'h200000000:0;''')
    text = once(text, '"CASE1_RESERVATIONS");passed();',
                '"CASE1_RESERVATIONS");rejected_request(0,make_request(2,32768,64),MAP_MISS);passed();')
    text = once(text, 'd=make_request(4,12288);',
                'd=make_request(4,12288);c.source_offset=16384;d.source_offset=20480;')
    text = once(text, 'cfg_valid=1;tick(8);check(!cfg_ready,"CFG_CHANGED_WHILE_HELD");cfg_valid=0;',
                'saved_map=cfg_map;cfg_map.first.physical_base=64\'h100000;cfg_valid=1;tick(8);'
                'check(!cfg_ready,"CFG_CHANGED_WHILE_HELD");cfg_valid=0;cfg_map=saved_map;')
    assert 'dut.' not in text and '`GUARD' not in text
    return text


def prepare(root, out, netlist):
    # Reuse the original library isolation and simulation flow, only adding the
    # map type package and selecting the new public-port bench/top.
    import prepare_ntrpu_peer_shared_routed as base
    base.prepare(root, out, netlist)
    (out / 'tb_ntrpu_peer_shared.sv').write_text(public_bench(root), encoding='utf-8', newline='\n')
    path = out / 'run.tcl'; text = path.read_text(encoding='utf-8')
    line = '    add_files -fileset sim_1 [file join $root fpga experimental ntrpu_peer_read_pkg.sv]'
    text = once(text, line, line + '\n    add_files -fileset sim_1 [file join $root fpga experimental ntrpu_peer_region_pkg.sv]')
    path.write_text(text.replace('SHARED_ROUTED_EXPORTED', 'REGIONS_ROUTED_EXPORTED'), encoding='utf-8', newline='\n')
    path = out / 'COMPARISON.json'; meta = json.loads(path.read_text(encoding='utf-8'))
    meta.update(sole_dut='ntrpu_peer_regions_physical', mapped_regions=2, remap_generations=[1,2],
                native_hbm_controller=False, memory_response_model='In-order public AXI responder',
                generated_bench_sha256=hashlib.sha256((out/'tb_ntrpu_peer_shared.sv').read_bytes()).hexdigest())
    path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8',newline='\n')
    return meta
