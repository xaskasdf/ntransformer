# Exploratory route of one generic DDR channel, not V6 joint implementation.
proc ntrpu_ddr_route {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and PLACED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_ports]] != 154} {
        error "Unexpected part/port count"
    }
    if {[llength [get_ports -filter {PACKAGE_PIN == AM17}]]} { error "AM17 occupied" }
    if {[get_property CLOCK_DEDICATED_ROUTE [get_nets ntrpu_ref_raw_BUFGCE]] ne "BACKBONE"} {
        error "Missing dedicated clock backbone"
    }
    report_drc -checks [get_drc_checks DCIRST-1] -file [file join $output dci_only.rpt]
    if {[llength [get_drc_violations -quiet]]} { error "DCI violation" }
    # No added timing exceptions, delay assumptions, or severity changes.
    set failed [catch {route_design} message options]
    set stream [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output route_attempt.dcp]
    report_route_status -file [file join $output route.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output io.rpt]
    write_xdc [file join $output routed_candidate.xdc]
    if {$failed} { error "Routing rejected; reports/checkpoint preserved" }
    puts "NTRPU_DDR_ROUTE_EXPORTED; review timing/DRC/IO; generic channel only; no board approval"
    close_project
}
if {[catch {ntrpu_ddr_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
