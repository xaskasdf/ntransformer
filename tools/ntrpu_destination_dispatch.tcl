# Isolated single-clock frontend simulation and synthesis, no board queue/IP.
set root [file normalize [file join [file dirname [info script]] ..]]
proc dispatch_unit {root output} {
    if {[file exists $output]} {error "Expected new output directory"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project dispatch_unit [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $root fpga rtl ntrpu_destination_dispatch.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_destination_dispatch.sv]
    set_property top tb_ntrpu_destination_dispatch [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach depth {1 3 8} {
        set_property generic DEPTH=$depth [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project dispatch_unit.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output depth-${depth}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "DQ SUITE_PASS depth=$depth cases=8" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Dispatch unit failed at depth $depth"}
    }
    close_project
    create_project dispatch_synth [file join $output synthesis] -part xcvu5p-flva2104-2-i
    add_files [file join $root fpga rtl ntrpu_destination_dispatch.sv]
    synth_design -top ntrpu_destination_dispatch -part xcvu5p-flva2104-2-i -mode out_of_context
    create_clock -name source -period 5 [get_ports clk]
    report_utilization -file [file join $output utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing_synth.rpt]
    write_checkpoint [file join $output dispatch_synth.dcp]
    set f [open [file join $output structure.txt] w]
    puts $f "LATCHES [llength [get_cells -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]"
    close $f
    close_project
    puts NTRPU_DESTINATION_DISPATCH_EXPORTED
}
if {[llength $argv]!=1} {error "Expected new output directory"}
if {[catch {dispatch_unit $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
