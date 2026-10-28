# MSI/MSI-X application handshake; protocol peers only until XDMA integration.
set script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ffn_irq_control {arguments} {
    global script_dir
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {error "Expected sim|sim_corrupt|synth NEW_DIRECTORY"}
    lassign $arguments mode output
    if {$mode ni {sim sim_corrupt synth}} {error "Unsupported mode"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project ffn_irq_control [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_ffn_control ntrpu_irq_transport ntrpu_ffn_irq_control} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set_property top ntrpu_ffn_irq_control [get_filesets sources_1]
    if {$mode ne "synth"} {
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_ffn_irq_control.sv]]
        set_property top tb_ntrpu_ffn_irq_control [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach profile {0 1 2} {
            set options "-testplusarg PROFILE=$profile"
            if {$mode eq "sim_corrupt"} {append options " -testplusarg INJECT_EARLY"}
            set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project ffn_irq_control.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output profile-${profile}.log]
            set f [open $log r];set text [read $f];close $f
            if {[regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text] ||
                ![string match "*NTRPU_CIRQ_PASS cases=18 *" $text]} {
                error "IRQ bank failed profile=$profile"
            }
        }
        puts NTRPU_CIRQ_SUITE_RETURNED
    } else {
        set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name device_clk -period 5.000 [get_ports clk]}
        puts $f {create_clock -name host_clk -period 4.000 [get_ports host_clk]}
        close $f
        read_xdc [file join $output clocks.xdc]
        synth_design -mode out_of_context -top ntrpu_ffn_irq_control -part xcvu5p-flva2104-2-i
        if {[llength [get_clocks]]!=2} {error "Clock count"}
        set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
        set blackboxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
        if {[llength $latches] || [llength $blackboxes]} {error "Latches or blackboxes"}
        write_checkpoint [file join $output ffn_irq_control_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_clocks -file [file join $output clocks.rpt]
        report_clock_interaction -file [file join $output clock_interaction.rpt]
        report_cdc -details -file [file join $output cdc.rpt]
        report_bus_skew -file [file join $output bus_skew.rpt]
        report_exceptions -coverage -file [file join $output exceptions.rpt]
        report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        write_xdc -constraints all [file join $output applied.xdc]
        puts NTRPU_CIRQ_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {ntrpu_ffn_irq_control $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
