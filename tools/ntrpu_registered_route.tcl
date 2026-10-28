set root [file normalize [file join [file dirname [info script]] ..]]
proc route_checkpoint {root output threads stage} {
    set_param general.maxThreads $threads
    if {$stage ni {synth physopt}} {error "Unknown resume stage"}
    open_checkpoint [file join $output registered_dispatch_${stage}.dcp]
    if {$stage eq "synth"} {
        read_xdc [file join $root fpga constraints ntrpu_multibuffer_physical_ooc.xdc]
    }
    if {[llength [get_sites -quiet BUFGCE_X0Y60]]!=1} {error "Assumed parent clock site unavailable"}
    set_property HD.CLK_SRC BUFGCE_X0Y60 [get_ports clk]
    if {$stage eq "synth"} {
        opt_design
        place_design
        write_checkpoint [file join $output registered_dispatch_place.dcp]
        report_timing_summary -delay_type min_max -file [file join $output timing_place.rpt]
        phys_opt_design
        write_checkpoint [file join $output registered_dispatch_physopt.dcp]
        report_timing_summary -delay_type min_max -file [file join $output timing_physopt.rpt]
    }
    route_design
    write_checkpoint [file join $output registered_dispatch_route.dcp]
    report_route_status -file [file join $output route_status.rpt]
    report_utilization -file [file join $output utilization_route.rpt]
    report_drc -file [file join $output drc_route.rpt]
    report_methodology -file [file join $output methodology_route.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing_route.rpt]
    report_timing -delay_type max -max_paths 5 -file [file join $output setup_paths.rpt]
    report_timing -delay_type min -max_paths 5 -file [file join $output hold_paths.rpt]
    write_verilog -mode funcsim [file join $output registered_dispatch_route.v]
    set f [open [file join $output physical_context.txt] w]
    puts $f "OOC clock and I/O budgets are provisional; no board timing signoff."
    puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks multibuffer_clk]]"
    puts $f "HD.CLK_SRC [get_property HD.CLK_SRC [get_ports clk]]"
    puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
    puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    close $f
    puts NTRPU_REGISTERED_DISPATCH_ROUTED
    close_design
}
if {[llength $argv]!=3} {error "Expected suite, thread count and resume stage"}
if {[catch {route_checkpoint $root [file normalize [lindex $argv 0]] [lindex $argv 1] [lindex $argv 2]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
