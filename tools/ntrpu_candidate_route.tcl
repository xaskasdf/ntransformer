# Route the preserved registered timing fixture; no RTL/XDC relaxation.
proc candidate_route {output threads} {
    set_param general.maxThreads $threads
    open_checkpoint [file join $output inputs candidate_place.dcp]
    if {[llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks sample_clk]]!=5.0} {
        error "Unexpected clock context"
    }
    if {[get_property LOC [get_cells parent_clock]] ne "BUFGCE_X0Y60"} {error "Parent BUFG moved"}
    write_xdc -constraints all [file join $output constraints_before.xdc]
    report_exceptions -coverage -file [file join $output exceptions_before.rpt]
    set failed [catch {route_design} message options]
    set f [open [file join $output route_command.txt] w]
    puts $f "RETURN_CODE $failed\nMESSAGE $message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    # A returning router always leaves a diagnostic checkpoint, even on error.
    write_checkpoint [file join $output candidate_route_attempt.dcp]
    report_route_status -file [file join $output route_status.rpt]
    report_drc -file [file join $output drc.rpt]
    set f [open [file join $output drc_violations.tsv] w]
    puts $f "name\tseverity"
    foreach violation [get_drc_violations -quiet] {
        puts $f "[get_property NAME $violation]\t[get_property SEVERITY $violation]"
    }
    close $f
    report_methodology -file [file join $output methodology.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_timing -from [all_registers] -to [all_registers] -delay_type max -max_paths 10 -file [file join $output internal_setup.rpt]
    report_timing -from [all_registers] -to [all_registers] -delay_type min -max_paths 10 -file [file join $output internal_hold.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    write_xdc -constraints all [file join $output constraints_after.xdc]
    write_verilog -mode funcsim [file join $output candidate_fixture_funcsim.v]
    write_verilog -cell core -mode funcsim -rename_top ntrpu_candidate_routed_core [file join $output candidate_core_funcsim.v]
    set f [open [file join $output metrics.txt] w]
    puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks sample_clk]]"
    puts $f "PARENT_CLOCK_LOC [get_property LOC [get_cells parent_clock]]"
    puts $f "CLOCKS [llength [get_clocks]]"
    puts $f "BUFGCE [llength [get_cells -hier -filter {REF_NAME == BUFGCE}]]"
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
    puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
    close $f
    if {$failed} {error "Router failed; diagnostic checkpoint retained"}
    puts NTRPU_CANDIDATE_ROUTE_EXPORTED
    close_design
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {2 4}} {error "Expected prepared output and thread count"}
if {[catch {candidate_route [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
