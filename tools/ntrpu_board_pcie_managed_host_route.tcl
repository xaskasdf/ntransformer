# Exploratory OOC implementation of the native-qualified cold transport/pool.
# Explicit assumed parent clock sites are not PCB clock or pin assignments.
proc route_managed_host {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected CONSTRAINED_DCP NEW_OUTPUT"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_dual_host_cold" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 5} {
        error "Unexpected checkpoint"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
        error "Unexpected blackbox or transparent storage"
    }
    # Do not let the inherited wildcard exception silently broaden.
    set chains {source_count_cdc source_reset_cdc source_up_cdc source_held_cdc admission_cdc}
    foreach reader {0 1} {
        foreach suffix {count_cdc reset_cdc up_cdc held_cdc ready_cdc} {
            lappend chains [format {memory[%d].%s} $reader $suffix]
        }
    }
    set expected {}
    foreach chain $chains {
        for {set stage 0} {$stage < 3} {incr stage} {
            lappend expected [format {core/receiver/startup/%s/release_ff_reg[%d]} $chain $stage]
        }
    }
    set releases [lsort [get_cells -quiet -hierarchical -filter {NAME =~ *release_ff_reg*}]]
    if {$releases ne [lsort $expected] || [llength $releases] != 45} {error "Reset endpoint inventory changed"}
    set f [open [file join $output reset_endpoints.txt] {WRONLY CREAT EXCL}]
    foreach cell $releases {
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0" ||
            [get_property ASYNC_REG $cell] ni {1 TRUE true}} {error "Unexpected reset primitive"}
        puts $f "$cell/CLR"
    }
    close $f
    write_xdc -constraints all [file join $output before_context.xdc]
    set f [open [file join $output clock_context.tsv] {WRONLY CREAT EXCL}]
    puts $f "clock\tport\tperiod\tassumed_site"
    set sites {}
    foreach {clock port period site} {
        source_clk source_clk 5.0 BUFGCE_X0Y60
        host_clk host_clk 4.0 BUFGCE_X0Y61
        ui0_clk {ui_clk[0]} 5.0 BUFGCE_X0Y62
        ui1_clk {ui_clk[1]} 5.0 BUFGCE_X0Y63
        mgmt_clk mgmt_clk 10.0 BUFGCE_X0Y64
    } {
        if {[get_property PERIOD [get_clocks $clock]] != $period} {error "Clock period changed: $clock"}
        set object [get_sites -quiet $site]
        if {[llength $object] != 1 || [get_property SITE_TYPE $object] ne "BUFGCE"} {error "Unavailable assumed clock source: $site"}
        lappend sites $site
        report_property -all $object -file [file join $output ${clock}_assumed_site.rpt]
        set_property HD.CLK_SRC $site [get_ports $port]
        puts $f "$clock\t$port\t$period\t$site"
    }
    close $f
    if {[llength [lsort -unique $sites]] != 5} {error "Parent clock sources overlap"}
    write_xdc -constraints all [file join $output with_context.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Diagnostic OOC implementation\n\nFive independent parent ports. Clock locations in clock_context.tsv are explicit assumptions, not placed board buffers. UI1 retains the simulated0.731ns phase."
    puts $f "No input/output delays, parent pblock or board pins. PCIe endpoint, MIG controllers/PHY, compute and four-channel board are outside this top. No physical timing or pinout approval."
    puts $f "Native cold-qualified RTL unchanged. Vendor constraints/waivers retained; inherited45-CLR exceptions exact-name/count/primitive checked. No additional false paths, multicycle exceptions or severity changes."
    puts $f "Protected FIFO internal diagnostics remain vendor-level evidence. Examine post-route DRC/timing/CDC rather than declaring them independently proven."
    close $f
    set_msg_config -id {Route 35-198} -limit 10000
    set failed [catch {
        opt_design
        place_design
        write_checkpoint [file join $output placed.dcp]
        report_timing_summary -report_unconstrained -file [file join $output placed_timing.rpt]
        route_design
    } message options]
    set f [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $f "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    write_checkpoint [file join $output route_attempt.dcp]
    write_verilog -mode funcsim [file join $output routed_funcsim.v]
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
    if {$failed} {error "Implementation attempt failed; partial checkpoint and reports retained"}
    puts NTRPU_MANAGED_HOST_ROUTE_EXPORTED
    close_project
}
if {[catch {route_managed_host $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
