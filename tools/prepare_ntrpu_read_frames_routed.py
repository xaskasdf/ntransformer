"""Exercise the unchanged routed netlist as the sole read DUT via public ports."""
import json
from pathlib import Path
from check_ntrpu_read_frames import sha

def prepare(root,out,netlist):
    bench=(root/'fpga/tb/tb_ntrpu_read_return.sv').read_text()
    assert bench.count('ntrpu_hbm_read_return dut(.*);')==1
    bench=bench.replace('ntrpu_hbm_read_return dut(.*);','ntrpu_read_return_physical dut(.*);')
    start=bench.index('    for(genvar f=0;f<2;f++)begin: frame_monitor')
    end=bench.index('    task automatic boot;',start);bench=bench[:start]+bench[end:]
    start=bench.index('    for(genvar w=0;w<2;w++)begin: window_monitor')
    end=bench.index('    initial begin #200000;',start);bench=bench[:start]+bench[end:]
    bench=bench.replace('    integer scenario=','    integer held_native_0,held_native_1;\n    integer scenario=',1)
    bench=bench.replace('        begin_case(2);','        begin_case(2);held_native_0=native_reads[0];held_native_1=native_reads[1];',1)
    old='wait(dut.channels[0].frame.bank_state[FRAMES-1]==2&&dut.channels[1].frame.bank_state[FRAMES-1]==2);'
    assert bench.count(old)==1
    bench=bench.replace(old,'wait(native_reads[0]==held_native_0+128*FRAMES&&native_reads[1]==held_native_1+128*FRAMES&&adapter_pending==0);')
    marker='    initial begin\n        $display("RF CONFIG'
    assert bench.count(marker)==1
    bench=bench.replace(marker,'    initial begin\n        #150;\n        $display("RF ROUTED_PUBLIC_ONLY");\n        $display("RF CONFIG')
    assert 'dut.channels' not in bench
    bench=bench.replace('endmodule','''
    always @(posedge mem_rst_n)$display("RN MEM_RESET %0d %0d",stamp(),epoch);
    always @(posedge started)$display("RN STARTED %0d %0d",stamp(),epoch);
endmodule''')
    (out/'tb_ntrpu_read_return.sv').write_text(bench)
    (out/'route_isolated.v').write_bytes(netlist.read_bytes())
    (out/'route_library.v').write_text('`uselib lib=read_return_routed lib=unisims_ver lib=secureip\n`include "route_isolated.v"\n`uselib\n')
    flow=(root/'tools/ntrpu_read_return.tcl').read_text()
    flow=flow[:flow.index('if {[llength $argv]!=3}')]
    flow=flow.replace('[file join $root fpga tb tb_ntrpu_read_return.sv]','[file join $output tb_ntrpu_read_return.sv]')
    flow=flow.replace('    launch_simulation;run all;close_sim','''    add_files -fileset sim_1 [file join $output route_library.v]
    set_property library read_return_routed [get_files [file join $output route_library.v]]
    launch_simulation;run all;close_sim''')
    flow+='\nif {[catch {simulate_read_return {'+str(root)+'} {'+str(out)+'} 0 0} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n'
    (out/'run.tcl').write_text(flow)
    (out/'COMPARISON.json').write_text(json.dumps(dict(netlist_sha256=sha(netlist),isolated_sha256=sha(out/'route_isolated.v'),
        source_bench_sha256=sha(root/'fpga/tb/tb_ntrpu_read_return.sv'),sole_read_dut='ntrpu_read_return_physical',
        public_ports_only=True,global_reset_wait_ns=150,sdf=False,force=False,netlist_edited=False,
        vendor_primitives_edited=False,reset_phase_guard=False,cycle_equivalence=False),indent=2))
