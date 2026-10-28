# Standalone AXI activation loader: real reader, range validation, cancellation.
set script_dir [file dirname [file normalize [info script]]]
proc activation_loader {arguments} {
    global script_dir
    if {[llength $arguments] != 2} { error "Expected sim|synth NEW_DIRECTORY" }
    lassign $arguments mode output
    if {$mode ni {sim synth}} { error "Invalid mode" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output exists" }
    file mkdir $output
    create_project activation_loader [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_axi_read_master ntrpu_activation_loader} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set_property top ntrpu_activation_loader [get_filesets sources_1]
    if {$mode eq "sim"} {
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_activation_loader.sv]]
        set_property top tb_ntrpu_activation_loader [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach seed {1 7 42} {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project activation_loader.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match "*NTRPU_ACTIVATION_PASS cases=32 *" $text] ||
                ![string match "*NTRPU_ACTIVATION_SEED seed=$seed\n*" $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} { error "Activation validation failed seed=$seed" }
        }
        puts "NTRPU_ACTIVATION_SUITE_RETURNED cases=96"
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc [file join $output clock.xdc]
        synth_design -top ntrpu_activation_loader -mode out_of_context -part xcvu5p-flva2104-2-i
        write_checkpoint [file join $output activation_loader_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
        puts NTRPU_ACTIVATION_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {activation_loader $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
