set script_dir [file dirname [file normalize [info script]]]
proc netlist_suite {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    exec python3 -B [file join $script_dir prepare_ntrpu_composition_netlist.py] [file join $output bench]
    create_project composition_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file normalize [file join $script_dir .. evidence routed_core_funcsim.v]]
    set_property top ntrpu_composition_core [get_filesets sources_1]
    add_files -fileset sim_1 [file join $output bench netlist_tb.sv]
    set_property top tb_ntrpu_peer_multibuffer_credits [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach profile {0 1 2} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg PROFILE=$profile"] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project composition_netlist.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output profile-${profile}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "MEMORY_SUITE_PASS profile=$profile" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Routed functional regression failed profile=$profile"}
    }
    close_project
    puts NTRPU_COMPOSITION_NETLIST_EXPORTED
}
if {[catch {netlist_suite $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
