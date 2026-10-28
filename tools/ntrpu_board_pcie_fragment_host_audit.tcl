# Read-only audit: distinguish both OOC parent-clock gaps from data routing.
proc ntrpu_fragment_sync_inventory {path} {
    set stream [open $path {WRONLY CREAT EXCL}]
    puts $stream "cell\tref\tasync_reg\td_pin_nets\tq_pin_nets\td_fanout_pins"
    foreach cell [lsort [get_cells -hierarchical -filter {ASYNC_REG == 1}]] {
        set dn [get_nets -quiet -of_objects [get_pins -quiet ${cell}/D]]
        set qn [get_nets -quiet -of_objects [get_pins -quiet ${cell}/Q]]
        set loads [get_pins -quiet -leaf -of_objects $qn -filter {DIRECTION == IN}]
        puts $stream "$cell\t[get_property REF_NAME $cell]\t[get_property ASYNC_REG $cell]\t$dn\t$qn\t[lsort $loads]"
    }
    close $stream
}
proc ntrpu_fragment_host_route_audit {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP ROUTED_DCP NEW_DIRECTORY"
    }
    lassign $arguments synthesis checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $synthesis [file join $output synthesis.dcp]
    open_checkpoint [file join $output synthesis.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_host"} {error "Wrong synthesis top"}
    ntrpu_fragment_sync_inventory [file join $output synthesis_sync.tsv]
    close_project
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    ntrpu_fragment_sync_inventory [file join $output routed_sync.tsv]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_host" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property HD.CLK_SRC [get_ports clk]] ne "BUFGCE_X0Y60" ||
        [get_property HD.CLK_SRC [get_ports host_clk]] ne "BUFGCE_X0Y61" ||
        [llength [get_clocks]] != 2} { error "Unexpected input context" }
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
    set blackboxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
    set stream [open [file join $output storage_audit.txt] {WRONLY CREAT EXCL}]
    puts $stream "TRANSPARENT_STORAGE_CELLS [llength $latches]\nBLACKBOX_CELLS [llength $blackboxes]"
    close $stream
    if {[llength $latches] || [llength $blackboxes]} { error "Unexpected storage/blackbox" }
    set routes [dict create]
    foreach clock {clk host_clk} {
        set route [get_property ROUTE [get_nets $clock]]
        if {![string match -nocase *gap* $route]} { error "Expected OOC parent gap: $clock" }
        dict set routes $clock $route
        set stream [open [file join $output ${clock}_route.txt] {WRONLY CREAT EXCL}]
        puts $stream $route
        close $stream
        report_property -all [get_nets $clock] -file [file join $output ${clock}_net.rpt]
        report_route_status -of_objects [get_nets $clock] -file [file join $output ${clock}_route.rpt]
    }
    set stream [open [file join $output partial_clock_aliases.tsv] {WRONLY CREAT EXCL}]
    puts $stream "net\tstatus\tmatching_parent_clock"
    set counts [dict create]
    set partial_counts [dict create clk 0 host_clk 0]
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        dict incr counts $status
        if {$status ne "PARTIAL"} { continue }
        set matches {}
        set route [get_property ROUTE $net]
        dict for {clock canonical} $routes {
            if {[string equal $route $canonical]} { lappend matches $clock }
        }
        if {[llength $matches] != 1} { error "Unexplained PARTIAL net: $net" }
        set clock [lindex $matches 0]
        dict incr partial_counts $clock
        puts $stream "$net\t$status\t$clock"
    }
    close $stream
    set stream [open [file join $output audit_totals.txt] {WRONLY CREAT EXCL}]
    puts $stream "PARTIAL_COUNTS $partial_counts\nROUTE_STATUS_COUNTS $counts"
    close $stream
    foreach clock {clk host_clk} {
        if {[dict get $partial_counts $clock] == 0} { error "No partial aliases: $clock" }
    }
    set stream [open [file join $output port_context.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\tdirection\tHD.CLK_SRC\tHD.PARTPIN_LOCS"
    foreach port [get_ports] {
        puts $stream "$port\t[get_property DIRECTION $port]\t[get_property HD.CLK_SRC $port]\t[get_property HD.PARTPIN_LOCS $port]"
    }
    close $stream
    report_route_status -file [file join $output route.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -warn_on_violation -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    puts "NTRPU_FRAGMENT_HOST_ROUTE_AUDIT_EXPORTED; read-only, both parent clock gaps classified without changing constraints"
    close_project
}
if {[catch {ntrpu_fragment_host_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
