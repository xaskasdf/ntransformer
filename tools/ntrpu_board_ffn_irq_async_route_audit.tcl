# Read-only audit: distinguish both OOC parent-clock gaps from data routing.
proc ntrpu_ffn_irq_async_route_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected ROUTED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_async_fabric" ||
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
    # These payload/send registers intentionally survive application reset.
    # Inspect physical configuration INIT independently after implementation.
    set prefix fabric/device/with_irq.control/transport
    set registers [get_cells -quiet -hierarchical -filter {NAME =~ *with_irq.control/transport/event_payload_reg* || NAME =~ *with_irq.control/transport/service_payload_reg* || NAME =~ *with_irq.control/transport/event_send_reg || NAME =~ *with_irq.control/transport/service_send_reg}]
    if {[llength $registers] != 131} { error "Transport storage count: [llength $registers]" }
    set stream [open [file join $output initialization.tsv] {WRONLY CREAT EXCL}]
    puts $stream "cell\tref_name\tinit"
    foreach cell [lsort $registers] {
        puts $stream "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]"
        if {[get_property REF_NAME $cell] ne "FDRE" || [get_property INIT $cell] ne "1'b0"} { error "Transport configuration initialization: $cell" }
    }
    close $stream
    foreach suffix {event_send_reg service_send_reg} {
        if {[llength [get_cells -quiet $prefix/$suffix]] != 1} { error "Missing send register: $suffix" }
    }
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
    # TIMING-37: record the real endpoints of every AXI-Lite R bundle source.
    # A fanout diagnostic is retained, never converted into a new exception.
    set cells [get_cells -quiet -hierarchical -filter {NAME =~ *clock_conv_lite_resp_r/handshake/dest_hsdata_ff_reg*}]
    if {![llength $cells]} { error "Read-response capture registers missing" }
    set stream [open [file join $output axil_read_fanout.tsv] {WRONLY CREAT EXCL}]
    puts $stream "capture_pin\tdriver\tendpoint_count\tendpoints"
    foreach cell $cells {
        set dst [get_pins $cell/D]
        set nets [get_nets -segments -of_objects $dst]
        set drivers [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
        if {[llength $drivers] != 1} { error "Unexpected capture driver: $dst" }
        set endpoints [lsort [all_fanout -flat -endpoints_only -from $drivers]]
        puts $stream "$dst\t$drivers\t[llength $endpoints]\t[join $endpoints {;}]"
    }
    close $stream
    report_route_status -file [file join $output route.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -warn_on_violation -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    puts "NTRPU_FFN_ASYNC_ROUTE_AUDIT_EXPORTED; read-only, both parent gaps, read-response fanout and131 INIT registers recorded"
    close_project
}
if {[catch {ntrpu_ffn_irq_async_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
