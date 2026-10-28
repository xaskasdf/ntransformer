# V4 exploratory route of one generic DDR channel plus serial PCIe.
# Not the V5/V6 four-channel implementation or a DMA-to-DDR datapath.
proc ntrpu_interface_route {arguments} {
    if {[llength $arguments] ni {2 3} || [llength [get_projects -quiet]]} {
        error "Expected INPUT_DCP NEW_DIRECTORY ?route|postroute-explore|preroute-explore?"
    }
    lassign $arguments checkpoint output mode
    if {$mode eq ""} { set mode route }
    if {$mode ni {route postroute-explore preroute-explore}} { error "Unknown mode" }
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
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "Unresolved black boxes"
    }
    if {[get_property CLOCK_DEDICATED_ROUTE [get_nets u_ddr/ntrpu_ref_raw_BUFGCE]] ne "BACKBONE"} {
        error "Missing dedicated clock backbone"
    }
    report_drc -checks [get_drc_checks {BIVC-1 DCIRST-1}] -file [file join $output precheck.rpt]
    if {[llength [get_drc_violations -quiet]]} { error "Targeted prerequisite DRC failed" }
    # No added delays, timing exceptions, waivers, or severity changes.
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    if {$mode eq "postroute-explore"} {
        # UG835 2026.1: post-route phys_opt reroutes its modified nets.
        # The input must already be fully routed; Vivado enforces this.
        report_route_status -file [file join $output input_route.rpt]
        report_timing_summary -report_unconstrained -file [file join $output input_timing.rpt]
        help phys_opt_design
        set failed [catch {phys_opt_design -directive Explore} message options]
    } elseif {$mode eq "preroute-explore"} {
        # Alternative starts from the original placed checkpoint; it must
        # not inherit an unsuccessful post-route solution as its baseline.
        report_timing_summary -report_unconstrained -file [file join $output input_timing.rpt]
        set failed [catch {
            phys_opt_design -directive Explore
            write_checkpoint [file join $output pre_route_optimized.dcp]
            report_timing_summary -report_unconstrained -file [file join $output pre_route_timing.rpt]
            route_design
        } message options]
    } else {
        set failed [catch {route_design} message options]
    }
    set stream [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $stream "mode=$mode\nreturn_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    write_checkpoint [file join $output route_attempt.dcp]
    report_route_status -file [file join $output route.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output io.rpt]
    write_xdc [file join $output routed_candidate.xdc]
    if {$failed} { error "Combined routing rejected; reports/checkpoint preserved" }
    puts "NTRPU_INTERFACE_ROUTE_EXPORTED; review timing/DRC/IO; generic one-channel probe only"
    close_project
}
if {[catch {ntrpu_interface_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
