# Exploratory OOC placement/routing of the validated local SwiGLU/Q8 block checkpoint.
# The explicit clock site is a provisional context, not a board assignment.
proc ntrpu_swiglu_block_route {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP ASSUMED_BUFGCE_SITE NEW_DIRECTORY"
    }
    lassign $arguments checkpoint clock_site output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    report_property -all [current_design] -file [file join $output input_design.rpt]
    report_clocks -file [file join $output input_clocks.rpt]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property TOP [current_design]] ne "ntrpu_swiglu_q8_block" ||
        [llength [get_ports -quiet {ap_start ap_done ap_ready gate_ce0 up_ce0 record_r_we0}]] != 6 ||
        [llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks ap_clk]] != 5.0} {
        error "Unexpected checkpoint or clock"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "Unresolved black boxes"
    }
    set site [get_sites -quiet $clock_site]
    if {[llength $site] != 1 || [get_property SITE_TYPE $site] ne "BUFGCE"} {
        error "Expected a single available BUFGCE site"
    }
    help place_design
    help route_design
    report_property -all $site -file [file join $output assumed_clock_site.rpt]
    set_property HD.CLK_SRC $clock_site [get_ports ap_clk]
    set stream [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $stream "# Exploratory OOC context\n\nClock: 5 ns, assumed external buffer $clock_site."
    puts $stream "HD.CLK_SRC defines an assumed parent clock location (AMD UG905), not an instantiated board clock."
    puts $stream "No external input/output delays, pin locations or parent pblock have been supplied."
    puts $stream "Internal routing/timing only; port paths and final parent clock require integration and reimplementation."
    puts $stream "No false paths, multicycle exceptions, severity changes or waivers are added."
    close $stream
    set failed [catch {
        opt_design
        place_design
        write_checkpoint [file join $output placed.dcp]
        report_timing_summary -report_unconstrained -file [file join $output placed_timing.rpt]
        route_design
    } message options]
    set stream [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output route_attempt.dcp]
    report_route_status -file [file join $output route.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    write_xdc [file join $output implemented.xdc]
    if {$failed} { error "OOC implementation failed; diagnostic checkpoint and reports retained" }
    puts "NTRPU_SWIGLU_BLOCK_ROUTE_EXPORTED; exploratory OOC, parent clock and boundary qualification pending"
    close_project
}
if {[catch {ntrpu_swiglu_block_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
