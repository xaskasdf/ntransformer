"""Derive the six shared-source cases using public ports of an untouched netlist."""
import hashlib
import json


def public_bench(root):
    text = (root / 'fpga/tb/tb_ntrpu_peer_shared.sv').read_text()
    assert text.count('ntrpu_peer_read_shared dut(') == 1
    text = text.replace('ntrpu_peer_read_shared dut(', 'ntrpu_peer_shared_physical dut(')
    lines = text.splitlines()
    for kind in ('cmd', 'cpl'):
        index = next(i for i, s in enumerate(lines) if f'if(dut.{kind}_valid' in s)
        assert '$display(' in lines[index + 1]
        del lines[index:index + 2]
    text = '\n'.join(lines) + '\n'
    old = '    `define GUARD dut.guard\n    `include "ntrpu_peer_shared_monitor.svh"\n    `undef GUARD'
    assert text.count(old) == 1
    text = text.replace(old, (root / 'fpga/tb/ntrpu_peer_shared_public.svh').read_text())
    marker = '    initial begin\n        boot(1);'
    assert text.count(marker) == 1
    text = text.replace(marker, '    initial begin\n        #150;\n        $display("SP PUBLIC_ONLY");\n        boot(1);')
    assert 'dut.' not in text and '`GUARD' not in text
    text = text.replace('SG PASS ', 'SP PASS ').replace('SG ALL_PASS mode=full cases=6', 'SP ALL_PASS cases=6')
    return text


def prepare(root, out, netlist):
    text = public_bench(root)
    (out / 'tb_ntrpu_peer_shared.sv').write_text(text)
    (out / 'route_isolated.v').write_bytes(netlist.read_bytes())
    (out / 'route_library.v').write_text('`uselib lib=shared_routed lib=unisims_ver lib=secureip\n`include "route_isolated.v"\n`uselib\n')
    (out / 'run.tcl').write_text('''set output [file normalize [lindex $argv 0]]
set root [file normalize [lindex $argv 1]]
if {[catch {
    set_param general.maxThreads 2
    create_project shared_route [file join $output project] -part xcvu47p-fsvh2892-2-e
    add_files -fileset sim_1 [file join $root fpga experimental ntrpu_peer_read_pkg.sv]
    add_files -fileset sim_1 [file join $output tb_ntrpu_peer_shared.sv]
    add_files -fileset sim_1 [file join $output route_library.v]
    set_property library shared_routed [get_files [file join $output route_library.v]]
    set_property include_dirs $output [get_filesets sim_1]
    set_property top tb_ntrpu_peer_shared [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    file copy [file join $output project shared_route.sim sim_1 behav xsim simulate.log] [file join $output simulation.log]
    set f [open [file join $output simulation.log]];set log [read $f];close $f
    if {[string first {SP ALL_PASS cases=6} $log]<0 || [regexp {Fatal:|SG_FAIL|SG_WATCHDOG} $log]} {error "Shared public regression failed"}
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts SHARED_ROUTED_EXPORTED
exit 0
''')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(netlist_sha256=sha(netlist), isolated_sha256=sha(out / 'route_isolated.v'),
                  source_bench_sha256=sha(root / 'fpga/tb/tb_ntrpu_peer_shared.sv'),
                  sole_dut='ntrpu_peer_shared_physical', public_ports_only=True, netlist_edited=False,
                  internal_forward_and_local_cpl_observed=False, sdf=False, force=False,
                  cycle_equivalence=False, global_reset_wait_ns=150)
    (out / 'COMPARISON.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
