# V4 exploratory placement of one generic DDR channel and serial PCIe.
# Does not qualify a DIMM, connect DMA to DDR, or implement four channels.
proc ntrpu_interface_place {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected COMBINED_1V2_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_ports]] != 221 ||
        [get_property PACKAGE_PIN [get_ports pcie_sys_rst_n]] ne "AM17" ||
        [get_property IOSTANDARD [get_ports pcie_sys_rst_n]] ne "LVCMOS12"} {
        error "Unexpected combined interface configuration"
    }
    report_drc -checks [get_drc_checks {BIVC-1 DCIRST-1}] -file [file join $output precheck.rpt]
    if {[llength [get_drc_violations -quiet]]} { error "Targeted prerequisite DRC failed" }
    # Clear inherited partial DDR routing only in this new working copy.
    route_design -unroute
    opt_design
    write_checkpoint [file join $output optimized.dcp]
    set failed [catch {place_design} message options]
    set stream [open [file join $output placement_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output placement_attempt.dcp]
    report_drc -file [file join $output drc.rpt]
    report_io -file [file join $output io.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -file [file join $output cdc.rpt]
    write_xdc [file join $output placed_candidate.xdc]
    if {$failed} { error "Combined placement rejected; evidence preserved" }
    puts "NTRPU_INTERFACE_PLACEMENT_EXPORTED; review timing/DRC; generic one-channel probe only"
    close_project
}
if {[catch {ntrpu_interface_place $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
