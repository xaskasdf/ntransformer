# Checked output conversion, with immutable independently generated vectors.
set script_dir [file dirname [file normalize [info script]]]
proc fp32_converter {arguments} {
    global script_dir
    if {[llength $arguments] != 3} { error "Expected sim|synth|netlist FIXTURE_OR_SYNTH_OUTPUT NEW_DIRECTORY" }
    lassign $arguments mode input output
    if {$mode ni {sim synth netlist}} { error "Invalid mode" }
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} { error "Output exists" }
    file mkdir $output
    if {$mode eq "netlist"} {
        file copy [file join $input fp32_converter_synth.dcp] [file join $output input_synth.dcp]
        file copy [file join $input fixture] [file join $output fixture]
        open_checkpoint [file join $output input_synth.dcp]
        write_verilog -mode funcsim [file join $output fp32_converter_netlist.v]
        close_design
    } else {file copy $input [file join $output fixture]}
    create_project fp32_converter [file join $output project] -part xcvu5p-flva2104-2-i
    if {$mode eq "netlist"} {add_files [file join $output fp32_converter_netlist.v]} else {
        add_files [file normalize [file join $script_dir .. fpga rtl ntrpu_fp64_to_fp32.sv]]
    }
    set_property top ntrpu_fp64_to_fp32 [get_filesets sources_1]
    if {$mode ne "synth"} {
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_fp64_to_fp32.sv]]
        set_property top tb_ntrpu_fp64_to_fp32 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach seed {1 7 42} {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed -testplusarg FIXTURE=[file join $output fixture]"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project fp32_converter.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match "*NTRPU_FPCAST_PASS count=51928 accepted=51928 resets=4 *" $text] ||
                ![string match "*NTRPU_FPCAST_SEED seed=$seed count=51928\n*" $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Converter validation failed seed=$seed"}
        }
        puts "NTRPU_FPCAST_SUITE_RETURNED mode=$mode count=155784"
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc [file join $output clock.xdc]
        synth_design -top ntrpu_fp64_to_fp32 -mode out_of_context -part xcvu5p-flva2104-2-i
        write_checkpoint [file join $output fp32_converter_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
        puts NTRPU_FPCAST_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {fp32_converter $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
