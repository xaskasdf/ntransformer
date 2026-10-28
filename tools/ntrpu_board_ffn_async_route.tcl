# Exploratory OOC placement/routing of the validated memory-to-memory FFN checkpoint.
# The explicit clock site is a provisional context, not a board assignment.
proc ntrpu_ffn_async_route {arguments} {
    if {[llength $arguments] != 4 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP ASSUMED_DEVICE_BUFGCE_SITE ASSUMED_HOST_BUFGCE_SITE NEW_DIRECTORY"
    }
    lassign $arguments checkpoint clock_site host_clock_site output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    report_property -all [current_design] -file [file join $output input_design.rpt]
    report_clocks -file [file join $output input_clocks.rpt]
    set top [get_property TOP [current_design]]
    set required_ports {s_axil_awvalid s_axil_wvalid s_axil_rvalid s_host_axi_awvalid s_host_axi_wvalid s_host_axi_bready s_host_axi_arvalid s_host_axi_rready m_axi_awvalid m_axi_wvalid m_axi_bready irq host_clk host_rst_n clk rst_n}
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        $top ne "ntrpu_ffn_async_fabric" ||
        [llength [get_ports -quiet $required_ports]] != [llength $required_ports] ||
        [llength [get_clocks]] != 2 || [get_property PERIOD [get_clocks clk]] != 5.0 ||
        [get_property PERIOD [get_clocks host_clk]] != 4.0} {
        error "Unexpected checkpoint or clocks"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "Unresolved black boxes"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]]} {
        error "Unexpected transparent storage"
    }
    if {$clock_site eq $host_clock_site} { error "Clock sites must differ" }
    set host_site [get_sites -quiet $host_clock_site]
    if {[llength $host_site] != 1 || [get_property SITE_TYPE $host_site] ne "BUFGCE"} {
        error "Expected a single available host BUFGCE site"
    }
    report_property -all $host_site -file [file join $output assumed_host_clock_site.rpt]
    set_property HD.CLK_SRC $host_clock_site [get_ports host_clk]
    set site [get_sites -quiet $clock_site]
    if {[llength $site] != 1 || [get_property SITE_TYPE $site] ne "BUFGCE"} {
        error "Expected a single available BUFGCE site"
    }
    help place_design
    help route_design
    report_property -all $site -file [file join $output assumed_clock_site.rpt]
    set_property HD.CLK_SRC $clock_site [get_ports clk]
    set stream [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $stream "# Exploratory OOC context\n\nDevice clock: 5 ns, assumed external buffer $clock_site. Host: 4 ns, assumed external buffer $host_clock_site."
    puts $stream "HD.CLK_SRC defines an assumed parent clock location (AMD UG905), not an instantiated board clock."
    puts $stream "No external input/output delays, pin locations or parent pblock have been supplied."
    puts $stream "Internal routing/timing only; port paths and final parent clock require integration and reimplementation."
    puts $stream "Vendor CDC XDC and waivers are retained. No new false paths, multicycle exceptions, severity changes or user waivers are added."
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
    report_cdc -details -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_bus_skew -warn_on_violation -file [file join $output bus_skew.rpt]
    write_xdc -constraints all [file join $output implemented.xdc]
    set stream [open [file join $output clock_context.tsv] {WRONLY CREAT EXCL}]
    puts $stream "clock\tperiod\tport\tassumed_source"
    foreach clock {clk host_clk} {
        puts $stream "$clock\t[get_property PERIOD [get_clocks $clock]]\t$clock\t[get_property HD.CLK_SRC [get_ports $clock]]"
    }
    close $stream
    if {$failed} { error "OOC implementation failed; diagnostic checkpoint and reports retained" }
    puts "NTRPU_FFN_ASYNC_ROUTE_EXPORTED; exploratory OOC, two parent clocks and boundary qualification pending"
    close_project
}
if {[catch {ntrpu_ffn_async_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
