"""Derive a public-port bench; do not substitute or edit the routed DUT."""
from pathlib import Path
import hashlib
import json


def prepare(root, out, netlist):
    text=(root/'fpga/tb/tb_ntrpu_peer_read.sv').read_text()
    start=text.index('        if(!PAIR)begin: separate')
    end=text.index('        assign arready[l]',start)
    text=text[:start]+text[end:]
    text=text.replace('ntrpu_peer_read_pair dut(', 'ntrpu_peer_read_physical dut(',1)
    start=text.index('        assign cmd_valid=dut.cmd_valid;')
    end=text.index("    // FULL=0's BFM",start)
    text=text[:start]+'''        for(genvar l=0;l<2;l++)begin
            assign araddr[l]=h_araddr[l*34+:34];
            assign arlen[l]={4'b0,h_arlen[l*4+:4]};
        end
    end
'''+text[end:]
    text=text.replace('commands[0]','nw[0]')
    lines=text.splitlines()
    lines=[line for line in lines if not any(token in line for token in
           ('if(held_cmd[l])','held_cmd[l]=cmd_valid','if(cpl_valid[l] && cpl_ready[l])'))]
    text='\n'.join(lines)+'\n'
    start=text.index('                if(cmd_valid[l] && cmd_ready[l])begin')
    end=text.index('                if(dma_valid[l] && dma_ready[l])begin',start)
    text=text[:start]+text[end:]
    marker='                    received[l][observed.identity.sequence_id]++;'
    assert text.count(marker)==1
    text=text.replace(marker,marker+'\n                    if(dma_last[l])local_completions[l]++;')
    marker='        request[lane]=v;request_valid[lane]=1;#0.001;while(!request_ready[lane])tick();tick();request_valid[lane]=0;'
    assert text.count(marker)==1
    text=text.replace(marker,marker+'''
        $display("PR REQUEST %0d %0d %0d %0d %0d %0d",scenario,epoch,lane,
            v.identity.sequence_id,expected_lease[lane].source_base+v.source_offset,v.bytes);''')
    text=text.replace('local_completions','payload_completions').replace('wait_local','wait_payload')
    text=text.replace('LOCAL_COMPLETION_RELEASED','PAYLOAD_RELEASED')
    marker='    initial begin\n        scenario=1;boot();'
    assert text.count(marker)==1
    text=text.replace(marker,'    initial begin\n        #150;\n        $display("PP ROUTED_PUBLIC_ONLY");\n        scenario=1;boot();')
    text=text.replace('PR ','PP ').replace('PR_FAIL','PP_FAIL').replace('PR_WATCHDOG','PP_WATCHDOG')
    assert 'dut.' not in text, 'Hierarchical observation survived'
    (out/'tb_ntrpu_peer_read.sv').write_text(text)
    (out/'route_isolated.v').write_bytes(netlist.read_bytes())
    (out/'route_library.v').write_text('`uselib lib=peer_read_routed lib=unisims_ver lib=secureip\n`include "route_isolated.v"\n`uselib\n')
    flow='''set output [file normalize [lindex $argv 0]]
set root [file normalize [lindex $argv 1]]
if {[catch {
    set_param general.maxThreads 2
    create_project peer_route [file join $output project] -part xcvu47p-fsvh2892-2-e -force
    add_files -fileset sim_1 [file join $root fpga experimental ntrpu_peer_read_pkg.sv]
    add_files -fileset sim_1 [file join $output tb_ntrpu_peer_read.sv]
    add_files -fileset sim_1 [file join $output route_library.v]
    set_property library peer_read_routed [get_files [file join $output route_library.v]]
    set_property include_dirs $output [get_filesets sim_1]
    set_property top tb_ntrpu_peer_read [get_filesets sim_1]
    set_property generic {FULL=1 PAIR=1} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    set source [file join $output project peer_route.sim sim_1 behav xsim simulate.log]
    file copy -force $source [file join $output simulation.log]
    set f [open [file join $output simulation.log]];set log [read $f];close $f
    if {[string first {PP ALL_PASS cases=21 full=1} $log]<0 || [string first {Fatal:} $log]>=0} {error "Public routed regression failed"}
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts PEER_READ_ROUTED_EXPORTED
exit 0
'''
    (out/'run.tcl').write_text(flow)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(netlist_sha256=digest(netlist),isolated_sha256=digest(out/'route_isolated.v'),
                source_bench_sha256=digest(root/'fpga/tb/tb_ntrpu_peer_read.sv'),
                sole_dut='ntrpu_peer_read_physical',public_ports_only=True,netlist_edited=False,
                command_and_local_cpl_observed=False,payload_boundary_observed=True,
                sdf=False,force=False,cycle_equivalence=False,global_reset_wait_ns=150)
    (out/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
