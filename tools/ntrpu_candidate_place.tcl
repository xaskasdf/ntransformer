# Paired physical diagnosis from preserved synthesis checkpoints. No routing.
proc place_pair {output selection threads} {
    set_param general.maxThreads $threads
    if {$selection eq "both"} {set variants {registered candidate}} else {set variants [list $selection]}
    foreach variant $variants {
        open_checkpoint [file join $output inputs ${variant}_synth.dcp]
        if {[get_property PERIOD [get_clocks sample_clk]] != 5.0} {error "Clock context changed"}
        if {[llength [get_cells -quiet parent_clock]]!=1} {error "Expected explicit parent BUFGCE"}
        if {[llength [get_sites -quiet BUFGCE_X0Y60]]!=1} {error "Clock site unavailable"}
        set_property LOC BUFGCE_X0Y60 [get_cells parent_clock]
        opt_design
        place_design
        write_checkpoint [file join $output ${variant}_place.dcp]
        report_clocks -file [file join $output ${variant}_clocks.rpt]
        report_utilization -file [file join $output ${variant}_utilization.rpt]
        report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output ${variant}_timing_place.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type max -max_paths 10 -file [file join $output ${variant}_internal_setup.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type min -max_paths 10 -file [file join $output ${variant}_internal_hold.rpt]
        report_high_fanout_nets -max_nets 10 -file [file join $output ${variant}_fanout.rpt]
        set f [open [file join $output ${variant}_context.txt] w]
        puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks sample_clk]]"
        puts $f "PARENT_CLOCK_LOC [get_property LOC [get_cells parent_clock]]"
        puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
        puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
        puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
        puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
        puts $f "ROUTE_NOT_REQUESTED; outer sampler ports are unqualified; no board timing claim."
        close $f
        close_design
    }
    puts NTRPU_CANDIDATE_PAIRED_PLACE_COMPLETE
}
if {[llength $argv]!=3} {error "Expected prepared output directory, variant and threads"}
if {[lindex $argv 1] ni {both registered candidate} || [lindex $argv 2] ni {2 4}} {error "Invalid placement selection"}
if {[catch {place_pair [file normalize [lindex $argv 0]] [lindex $argv 1] [lindex $argv 2]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
