# Static timing at lower frequencies on the exact routed fixture; no new P&R.
proc physical_identity {} {
    set cells [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}]
    set nets [get_nets -hierarchical]
    return [list $cells [get_property LOC $cells] [get_property BEL $cells] \
        $nets [get_property ROUTE_STATUS $nets] [get_property ROUTE $nets]]
}
proc period_audit {arguments} {
    if {[llength $arguments] < 2} {error "Expected ROUTED_FIXTURE_DCP NEW_DIRECTORY ?PERIOD ...?"}
    lassign $arguments checkpoint output
    set periods [lrange $arguments 2 end]
    if {![llength $periods]} {set periods {6.667 8.000}}
    if {[llength [lsort -unique $periods]] != [llength $periods]} {error "Duplicate period"}
    foreach period $periods {
        if {$period ni {5.715 6.000 6.667 8.000}} {error "Unreviewed audit period"}
    }
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    set_param general.maxThreads 2
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property TOP [current_design]] ne "ntrpu_multibuffer_physical_fixture" ||
        [llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks fixture_clk]] != 5.0} {
        error "Unexpected source fixture"
    }
    set identity [physical_identity]
    set f [open [file join $output summary.tsv] {WRONLY CREAT EXCL}]
    puts $f "period_ns\tworst_setup_ns\tworst_hold_ns\tphysical_identity_preserved"
    foreach period $periods {
        set folder [file join $output period_$period]
        file mkdir $folder
        create_clock -period $period -name fixture_clk [get_ports clk]
        set_clock_uncertainty -setup 0.100 [get_clocks fixture_clk]
        set_clock_uncertainty -hold 0.050 [get_clocks fixture_clk]
        report_timing_summary -report_unconstrained -file [file join $folder timing.rpt]
        check_timing -verbose -file [file join $folder check_timing.rpt]
        report_timing -delay_type max -max_paths 5 -path_type full_clock_expanded -file [file join $folder setup.rpt]
        report_timing -delay_type min -max_paths 5 -path_type full_clock_expanded -file [file join $folder hold.rpt]
        report_route_status -file [file join $folder route.rpt]
        report_clocks -file [file join $folder clocks.rpt]
        write_xdc -constraints all [file join $folder applied.xdc]
        if {[physical_identity] ne $identity} {error "Physical implementation changed during timing-only audit"}
        set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
        set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
        puts $f "$period\t$setup\t$hold\t1"
    }
    close $f
    puts "MULTIBUFFER_PERIOD_AUDIT_EXPORTED; same placement/routing, probe boundaries still unqualified"
    close_project
}
if {[catch {period_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
