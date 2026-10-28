set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc event_boundaries {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project event_boundaries $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_rc_proposal.sv ntrpu_pcie_rc_event_stage.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_event_boundaries.sv]
    set_property top tb_ntrpu_pcie_rc_event_boundaries [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_event_boundaries.txt]]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set f [open [file join $output event_boundaries.sim sim_1 behav xsim simulate.log] r]
    set text [read $f];close $f
    if {$caught || [string first "RC_BOUNDARY_SUITE_COMPLETE rows=370 phases=15 latency=2 native=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Event boundary simulation incomplete: $message"}
    close_project
    puts NTRPU_PCIE_RC_EVENT_BOUNDARIES_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {event_boundaries [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
