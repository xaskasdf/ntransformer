# Isolated automatic guard with actual XPM primitives and retained channels.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_guard_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project guard $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_reset_supervisor ntrpu_retained_channel ntrpu_host_loss_guard} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_host_loss_guard_r2.sv]
    set_property top tb_ntrpu_host_loss_guard_r2 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output guard.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output guard.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "GUARD_SUITE_PASS cases=10" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Host loss guard bank failed"}
    puts NTRPU_HOST_LOSS_GUARD_EXPORTED
    close_project
}
if {[catch {ntrpu_guard_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
