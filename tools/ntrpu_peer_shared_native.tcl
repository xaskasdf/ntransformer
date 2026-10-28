set root [file normalize [file join [file dirname [info script]] ..]]
set output [file normalize [lindex $argv 0]]
set_param general.maxThreads 1
if {[catch {
    file mkdir $output
    set f [open [file join $root tools ntrpu_read_return.tcl]];set text [read $f];close $f
    set end [string first {if {[llength $argv]!=3}} $text]
    if {$end<0} {error "Read configuration boundary changed"}
    eval [string range $text 0 [expr {$end-1}]]
    prepare_read_return $root $output
    foreach name {ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_admission ntrpu_peer_read_shared} {
        add_files [file join $root fpga experimental ${name}.sv]
    }
    add_files -fileset sim_1 [file join $root fixtures stub.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_peer_shared_native.sv]
    set_property include_dirs [list [file join $root fpga tb] [file join $root fixtures]] [get_filesets sim_1]
    set_property top tb_ntrpu_peer_shared_native [get_filesets sim_1]
    set_property generic {NATIVE=0} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    launch_simulation;run all;close_sim
    file copy [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log] [file join $output simulation.log]
    set f [open [file join $output simulation.log]];set log [read $f];close $f
    if {[string first {SN ALL_PASS native=0} $log]<0 || [regexp {Fatal:|SG_FAIL|SN_WATCHDOG} $log]} {error "Shared admission regression failed"}
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts PEER_SHARED_EXPORTED
exit 0
