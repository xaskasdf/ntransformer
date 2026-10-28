set script_dir [file dirname [file normalize [info script]]]
proc capacity_netlist {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected SYNTH_OUTPUT NEW_DIRECTORY"}
    lassign $arguments input output
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy [file join $input capacity_guard_synth.dcp] [file join $output input_synth.dcp]
    open_checkpoint [file join $output input_synth.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    write_verilog -mode funcsim [file join $output capacity_guard_netlist.v]
    close_design
    create_project capacity_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $output capacity_guard_netlist.v]
    set_property top ntrpu_axi_capacity_guard [get_filesets sources_1]
    add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_axi_capacity_guard.sv]]
    set_property top tb_ntrpu_axi_capacity_guard [get_filesets sim_1]
    set_property verilog_define NTRPU_NETLIST [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach seed {1 7 42} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project capacity_netlist.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output seed-${seed}.log]
        set f [open $log r];set text [read $f];close $f
        if {![regexp -line {^NTRPU_CAP_PASS cases=20 allowed=10 rejected=10 sw=429 mw=150 sr=429 mr=150 } $text] ||
            [regexp -nocase -line {^(Fatal:|ERROR:|FAIL |CRITICAL WARNING:)} $text]} {error "Capacity netlist failed seed=$seed"}
    }
    puts NTRPU_CAP_NETLIST_SUITE_PASS
    close_project
}
if {[catch {capacity_netlist $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
