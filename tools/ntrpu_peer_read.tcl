set root [file normalize [file join [file dirname [info script]] ..]]
set output [file normalize [lindex $argv 0]]
if {[catch {
    file mkdir $output
    set f [open [file join $root tools ntrpu_read_return.tcl]];set text [read $f];close $f
    set end [string first {if {[llength $argv]!=3}} $text]
    if {$end<0} {error "Read configuration boundary changed"}
    eval [string range $text 0 [expr {$end-1}]]
    prepare_read_return $root $output
    add_files [file join $root fpga experimental ntrpu_peer_read_pkg.sv]
    add_files [file join $root fpga experimental ntrpu_peer_read_lane.sv]
    add_files [file join $root fpga experimental ntrpu_peer_read_pair.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_peer_read.sv]
    set_property top tb_ntrpu_peer_read [get_filesets sim_1]
    set_property generic "FULL=1 PAIR=[lindex $argv 1]" [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    launch_simulation;run all;close_sim
    file copy [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log] [file join $output simulation.log]
    set f [open [file join $output simulation.log]];set log [read $f];close $f
    if {[string first {PR ALL_PASS cases=21 full=1} $log]<0 || [regexp {Fatal:|PR_FAIL|PR_WATCHDOG} $log]} {error "Peer read regression failed"}
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts PEER_READ_EXPORTED
exit 0
