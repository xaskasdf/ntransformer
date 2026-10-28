# Read-only audit of explicit gaps and port context in the OOC routed checkpoint.
proc ntrpu_ffn_memory_route_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} { error "Expected ROUTED_DCP NEW_DIRECTORY" }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    set top [get_property TOP [current_design]]
    if {$top ni {ntrpu_ffn_memory ntrpu_ffn_device ntrpu_ffn_fabric} ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property HD.CLK_SRC [get_ports clk]] ne "BUFGCE_X0Y60"} { error "Unexpected input context" }
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
    set stream [open [file join $output storage_audit.txt] {WRONLY CREAT EXCL}]
    puts $stream "TRANSPARENT_STORAGE_CELLS [llength $latches]"
    foreach cell $latches { puts $stream "$cell [get_property REF_NAME $cell]" }
    close $stream
    if {[llength $latches]} { error "Unexpected transparent storage in routed FFN" }
    help report_route_status
    report_property -all [get_nets clk] -file [file join $output clock_net.rpt]
    report_route_status -of_objects [get_nets clk] -file [file join $output clock_route.rpt]
    report_route_status -file [file join $output route.rpt]
    set all_nets [get_nets -hierarchical]
    set counts [dict create]
    # Avoid materializing the very large clock route once per hierarchical alias
    # in a bulk property list. Inspect PARTIAL segments one at a time and compare
    # each full route with the canonical clock route, saving that route only once.
    set canonical [get_property ROUTE [get_nets clk]]
    if {![string match -nocase *gap* $canonical]} { error "Expected parent clock gap" }
    set stream [open [file join $output canonical_clock_route.txt] {WRONLY CREAT EXCL}]
    puts $stream $canonical
    close $stream
    set stream [open [file join $output partial_clock_aliases.tsv] {WRONLY CREAT EXCL}]
    puts $stream "net\tstatus\tfull_route_matches_clock"
    set partial 0
    foreach net $all_nets status [get_property ROUTE_STATUS $all_nets] {
        dict incr counts $status
        if {$status eq "PARTIAL"} {
            if {![regexp -nocase {(clk|aclk)$} $net]} { error "Nonclock PARTIAL net: $net" }
            set route [get_property ROUTE $net]
            if {![string equal $route $canonical]} { error "Different partial route: $net" }
            incr partial
            puts $stream "$net\t$status\t1"
        }
    }
    close $stream
    if {$partial==0} { error "No PARTIAL clock segments found" }
    set stream [open [file join $output audit_totals.txt] {WRONLY CREAT EXCL}]
    puts $stream "PARTIAL_CLOCK_SEGMENTS $partial\nROUTE_STATUS_COUNTS $counts"
    puts $stream "Canonical clock contains a gap; every PARTIAL hierarchical segment has exactly its full route."
    puts $stream "Global report_route_status remains authoritative for total explicit gap nets."
    close $stream
    set stream [open [file join $output port_context.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\tdirection\tHD.CLK_SRC\tHD.PARTPIN_LOCS"
    foreach p [get_ports] {
        puts $stream "$p\t[get_property DIRECTION $p]\t[get_property HD.CLK_SRC $p]\t[get_property HD.PARTPIN_LOCS $p]"
    }
    close $stream
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    if {$top eq "ntrpu_ffn_fabric"} {
        puts "NTRPU_FFN_FABRIC_ROUTE_AUDIT_EXPORTED; read-only, no constraints or checkpoint modified"
    } elseif {$top eq "ntrpu_ffn_device"} {
        puts "NTRPU_FFN_HOST_ROUTE_AUDIT_EXPORTED; read-only, no constraints or checkpoint modified"
    } else {
        puts "NTRPU_FFN_MEMORY_ROUTE_AUDIT_EXPORTED; read-only, no constraints or checkpoint modified"
    }
    close_project
}
if {[catch {ntrpu_ffn_memory_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
