# Read-only audit of an isolated, terminal placed DDR checkpoint.
proc ntrpu_ddr_constraint_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and PLACED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected part" }
    set clocknet [get_nets ntrpu_ref_raw_BUFGCE]
    if {[get_property CLOCK_DEDICATED_ROUTE $clocknet] ne "BACKBONE"} { error "Missing clock constraint" }
    report_property -all -file [file join $output backbone_net.rpt] $clocknet
    set stream [open [file join $output clock_cells.tsv] {WRONLY CREAT EXCL}]
    puts $stream "cell\treference\tlocation\tclock_region"
    foreach cell [get_cells -hier -filter {REF_NAME =~ BUFG* || REF_NAME =~ MMCME* || REF_NAME =~ PLLE*}] {
        set site [get_sites -of_objects $cell]
        puts $stream "$cell\t[get_property REF_NAME $cell]\t[get_property LOC $cell]\t[get_clock_regions -of_objects $site]"
    }
    close $stream
    set stream [open [file join $output riu_pins.txt] {WRONLY CREAT EXCL}]
    foreach field {RIU_ADDR RIU_WR_DATA} {
        set pins [get_pins -hier -filter "NAME =~ */$field*"]
        if {![llength $pins]} { error "No $field pins" }
        puts $stream "$field count=[llength $pins]"
        foreach pin $pins { puts $stream $pin }
        # Original generated query is scoped to the MIG instance. Compare its
        # resolved endpoints to a hierarchical query without changing constraints.
        current_instance u_ntrpu_ddr_geometry/inst
        set original [get_pins */*/*/*/*/*.u_xiphy_control/xiphy_control/$field*]
        current_instance
        set expected [lsort [get_property NAME $pins]]
        set actual [lsort [get_property NAME $original]]
        puts $stream "original_query_count=[llength $original] same_endpoints=[expr {$expected eq $actual}]"
        if {$expected ne $actual} { error "Generated $field query endpoint mismatch" }
        report_timing -delay_type min -to $pins -max_paths 10 -file [file join $output ${field}_hold.rpt]
    }
    close $stream
    help report_exceptions
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    write_xdc [file join $output observed_constraints.xdc]
    puts "NTRPU_DDR_CONSTRAINT_AUDIT_EXPORTED; inspect coverage and diagnostics; no timing approval"
    close_project
}
if {[catch {ntrpu_ddr_constraint_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
