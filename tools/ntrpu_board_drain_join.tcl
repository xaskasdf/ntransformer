set root [file normalize [file join [file dirname [info script]] ..]]
proc run_join {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project join $output -part xcvu5p-flva2104-2-i
    add_files [file join $root fpga rtl ntrpu_drain_certificate_join.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_drain_certificate_join.sv]
    set_property top tb_ntrpu_drain_certificate_join [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    open_vcd [file join $output join.vcd]
    log_vcd [get_objects /tb_ntrpu_drain_certificate_join/*]
    run all
    close_vcd
    close_sim
    set log [file join $output join.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output join.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "JOIN_SUITE_PASS" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $text]} {error "Join simulation failed"}
    close_project
    puts NTRPU_DRAIN_JOIN_EXPORTED
}
if {[llength $argv]!=1} {exit 1}
if {[catch {run_join $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
