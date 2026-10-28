# Reuse a terminal synthesis checkpoint; validate generic DDR PHY placement.
# Usage: -tclargs SYNTHESIZED_DCP NEW_DIRECTORY ?backbone?
proc ntrpu_ddr_place {arguments} {
    if {[llength $arguments] ni {2 3} || [llength [get_projects -quiet]]} {
        error "Expected fresh process and SYNTHESIZED_DCP NEW_DIRECTORY ?backbone?"
    }
    lassign $arguments checkpoint output mode
    if {$mode ni {{} backbone}} { error "Unknown clock routing mode" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_ports]] != 154} {
        error "Unexpected part/port count"
    }
    foreach ref {DCIRESET STARTUPE3} {
        if {[llength [get_cells -hier -filter "REF_NAME == $ref"]] != 1} { error "Missing/duplicate $ref" }
    }
    if {[llength [get_ports -filter {PACKAGE_PIN == AM17}]]} { error "AM17 occupied" }
    report_drc -checks [get_drc_checks DCIRST-1] -file [file join $output dci_only.rpt]
    if {[llength [get_drc_violations -quiet]]} { error "DCIRST-1 unresolved" }
    set clocks [get_clocks -of_objects [get_pins ntrpu_ref_buffer/O]]
    if {[llength $clocks] != 1 || [get_property PERIOD $clocks] != 4.000} {
        error "Always-on sequencer reference is not one 4 ns clock: $clocks"
    }
    opt_design
    if {$mode eq "backbone"} {
        # The GCIO (bank 64) and MIG-selected MMCM (bank 66) share a CMT column.
        # UG912 BACKBONE/SAME_CMT_COLUMN retains dedicated global clock routes.
        # No_Buffer requires the user to provide this clock architecture constraint.
        set net [get_nets ntrpu_ref_raw_BUFGCE]
        if {[llength $net] != 1} { error "Unexpected inferred MIG clock net" }
        set drivers [get_cells -of_objects [get_pins -of_objects $net -filter {DIRECTION == OUT}]]
        if {[llength $drivers] != 1 || [get_property REF_NAME $drivers] ne "BUFGCE"} {
            error "MIG reference must be driven by one global clock buffer"
        }
        set_property CLOCK_DEDICATED_ROUTE BACKBONE $net
        report_property -all -file [file join $output backbone_net.rpt] $net
    }
    write_checkpoint [file join $output optimized.dcp]
    report_drc -file [file join $output optimized_drc.rpt]
    set failed [catch {place_design} message options]
    set stream [open [file join $output placement_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output placement_attempt.dcp]
    report_drc -file [file join $output placement_drc.rpt]
    report_io -file [file join $output io.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_utilization -file [file join $output utilization.rpt]
    if {$failed} { error "Placement rejected; diagnostic checkpoint/reports preserved" }
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_methodology -file [file join $output methodology.rpt]
    write_xdc [file join $output placed_candidate.xdc]
    puts "NTRPU_DDR_PLACEMENT_EXPORTED; inspect all diagnostics; no routing or exact DIMM approval"
    close_project
}
if {[catch {ntrpu_ddr_place $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
