# Diagnostic reset during the first synchronized hold-counter completion.
set script_dir [file dirname [file normalize [info script]]]
proc hold_retry {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project hold_retry [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    add_files [file join $script_dir .. fpga rtl ntrpu_reset_supervisor.sv]
    add_files -fileset sim_1 [file join $script_dir .. fpga tb tb_ntrpu_reset_hold_retry.sv]
    set_property top tb_ntrpu_reset_hold_retry [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set failed 0
    foreach profile {0 1} {
        set_property generic PROFILE=$profile [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project hold_retry.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output profile-$profile.log]
        set f [open $log r];set text [read $f];close $f
        if {![string match "*NTRPU_HOLD_RETRY_RETURNED profile=$profile attempts=8*" $text] ||
            [regexp -nocase {(^|\n)[ \t]*(Error:|Fatal:|FAIL )} $text]} {set failed 1}
    }
    if {$failed} {error "Hold completion retry diagnostic rejected; preserve logs"}
    puts NTRPU_HOLD_RETRY_SUITE_RETURNED
    close_project
}
if {[catch {hold_retry $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
