# V4 control experiment: implement the existing serial PCIe endpoint alone.
# Same AM17 LVCMOS12 candidate as the combined probe; no DDR or board approval.
proc ntrpu_pcie_implement {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected SERIAL_PCIE_SYNTH_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_ports]] != 67 ||
        [get_property PACKAGE_PIN [get_ports sys_rst_n]] ne "AM17" ||
        [get_property IOSTANDARD [get_ports sys_rst_n]] ne "LVCMOS18" ||
        [llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "Unexpected serial endpoint input"
    }
    foreach {type count} {GTYE4_CHANNEL 16 PCIE40E4 1} {
        if {[llength [get_cells -hierarchical -filter "REF_NAME == $type"]] != $count} {
            error "Unexpected primitive count: $type"
        }
    }
    # Match the combined probe's PERST candidate without modifying IP clocks,
    # timing exceptions, or the original serial endpoint checkpoint.
    set_property IOSTANDARD LVCMOS12 [get_ports sys_rst_n]
    report_io -file [file join $output input_io_1v2.rpt]
    report_drc -checks [get_drc_checks BIVC-1] -file [file join $output precheck.rpt]
    if {[llength [get_drc_violations -quiet]]} { error "BIVC precheck failed" }
    opt_design
    write_checkpoint [file join $output optimized.dcp]
    set failed [catch {place_design} message options]
    set stream [open [file join $output placement_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output placement_attempt.dcp]
    report_timing_summary -report_unconstrained -file [file join $output placement_timing.rpt]
    report_drc -file [file join $output placement_drc.rpt]
    if {$failed} { error "Isolated endpoint placement failed; evidence preserved" }
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
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output io.rpt]
    write_xdc [file join $output routed_candidate.xdc]
    if {$failed} { error "Isolated endpoint routing failed; evidence preserved" }
    puts "NTRPU_ISOLATED_PCIE_IMPLEMENTATION_EXPORTED; review timing/DRC; no DDR or board approval"
    close_project
}
if {[catch {ntrpu_pcie_implement $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
