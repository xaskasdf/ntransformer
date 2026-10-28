# Diagnostic implementation of a reviewed inventory DCP, with an assumed parent clock.
# No board pinout, endpoint, DDR, SDF or physical qualification is implied.
proc inventory_placement_reports {output prefix} {
    write_checkpoint [file join $output ${prefix}.dcp]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output ${prefix}_timing.rpt]
    check_timing -verbose -file [file join $output ${prefix}_check_timing.rpt]
    report_utilization -file [file join $output ${prefix}_utilization.rpt]
}

proc inventory_placement {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP NEW_OUTPUT"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    foreach command {open_checkpoint opt_design place_design route_design report_route_status report_timing_summary write_edif} {
        puts "NTRPU_HELP_BEGIN $command"
        help $command
        puts "NTRPU_HELP_END $command"
    }
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_inventory_join" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected checkpoint"}
    if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {
        error "Unexpected clock inventory/period"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
        error "Unexpected blackbox or transparent storage"
    }
    set ports [lsort [get_ports]]
    set f [open [file join $output ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection"
    foreach port $ports {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    write_xdc -constraints all [file join $output before_context.xdc]
    report_property -all [current_design] -file [file join $output input_design.rpt]
    report_property -all [get_clocks pcie_user_clk] -file [file join $output before_clock.rpt]
    report_property -all [get_ports clk] -file [file join $output before_clock_port.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output before_context_timing.rpt]
    report_exceptions -file [file join $output before_exceptions.rpt]
    set site BUFGCE_X0Y60
    set object [get_sites -quiet $site]
    if {[llength $object] != 1 || [get_property SITE_TYPE $object] ne "BUFGCE"} {
        error "Unavailable assumed parent clock source: $site"
    }
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected inherited parent clock source"}
    report_property -all $object -file [file join $output assumed_clock_site.rpt]
    set_property HD.CLK_SRC $site [get_ports clk]
    report_property -all [get_ports clk] -file [file join $output with_context_clock_port.rpt]
    write_xdc -constraints all [file join $output with_context.xdc]
    inventory_placement_reports $output with_context
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Inventory placement diagnostic\n\nOne4ns parent clock. HD.CLK_SRC BUFGCE_X0Y60 is an explicit assumption, not a placed parent buffer or board clock assignment."
    puts $f "Inherited zero input/output delays and clock uncertainty retained. No path exceptions, DRC severity changes, pipeline/capacity changes or resynthesis."
    puts $f "Default opt/place/route, maximum4 threads. Setup/hold violations remain failures. Parent clock gaps and internal routing must be reviewed separately."
    puts $f "No endpoint, DDR, PCB pins or native reset qualification. Exported netlist needs its own functional replay; no SDF timing simulation."
    close $f
    set phase optimization
    set failed [catch {
        opt_design
        inventory_placement_reports $output optimized
        set phase placement
        place_design
        inventory_placement_reports $output placed
        set phase routing
        route_design
        set phase completed
    } message options]
    set f [open [file join $output implementation_status.txt] {WRONLY CREAT EXCL}]
    puts $f "return_code=$failed\nphase=$phase\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    # Preserve terminal or partial state even if implementation did not finish.
    write_checkpoint [file join $output route_attempt.dcp]
    write_verilog -mode funcsim [file join $output routed_funcsim.v]
    write_edif [file join $output routed.edf]
    write_xdc -constraints all [file join $output implemented.xdc]
    set f [open [file join $output primitives.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    set reports_failed {}
    foreach {name command} {
        timing {report_timing_summary -report_unconstrained -max_paths 20}
        check_timing {check_timing -verbose}
        utilization {report_utilization}
        hierarchy {report_utilization -hierarchical}
        route {report_route_status}
        drc {report_drc}
        methodology {report_methodology}
        clocks {report_clocks}
        exceptions {report_exceptions}
        ignored_exceptions {report_exceptions -ignored}
        exception_summary {report_exceptions -summary}
    } {
        if {[catch {{*}$command -file [file join $output ${name}.rpt]} report_error]} {
            lappend reports_failed [list $name $report_error]
        }
    }
    set f [open [file join $output report_errors.txt] {WRONLY CREAT EXCL}]
    puts $f $reports_failed
    close $f
    set f [open [file join $output route_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus"
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {puts $f "$net\t$status"}
    close $f
    set clock_net [get_nets -of_objects [get_ports clk]]
    if {[llength $clock_net] != 1} {error "Nonunique parent clock net"}
    set clock_route [get_property ROUTE $clock_net]
    set f [open [file join $output parent_clock.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tclock\tperiod\tassumed_site\tnet\tstatus\troute"
    puts $f "clk\tpcie_user_clk\t[get_property PERIOD [get_clocks pcie_user_clk]]\t[get_property HD.CLK_SRC [get_ports clk]]\t$clock_net\t[get_property ROUTE_STATUS $clock_net]\t$clock_route"
    close $f
    set f [open [file join $output partial_routes.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus\tmatching_parent_route\troute"
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        if {$status ne "PARTIAL"} {continue}
        set net_route [get_property ROUTE $net]
        set matches [expr {$clock_route ne "" && $net_route eq $clock_route}]
        puts $f "$net\t$status\t$matches\t$net_route"
    }
    close $f
    if {[lsort [get_ports]] ne $ports || [llength [get_clocks]] != 1 ||
        [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {error "Interface/clock changed"}
    if {$failed || [llength $reports_failed]} {error "Implementation/report failure; partial evidence retained"}
    puts NTRPU_INVENTORY_PLACEMENT_DIAGNOSTIC_EXPORTED
    close_project
}
if {[catch {inventory_placement $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
