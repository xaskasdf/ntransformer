# Read-only audit of explicit gaps and port context in the OOC routed checkpoint.
proc ntrpu_gemv_quad_route_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} { error "Expected ROUTED_DCP NEW_DIRECTORY" }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property HD.CLK_SRC [get_ports clk]] ne "BUFGCE_X0Y60"} { error "Unexpected input context" }
    help report_route_status
    report_property -all [get_nets clk] -file [file join $output clock_net.rpt]
    report_route_status -of_objects [get_nets clk] -file [file join $output clock_route.rpt]
    report_route_status -file [file join $output route.rpt]
    set all_nets [get_nets -hierarchical]
    set counts [dict create]
    set stream [open [file join $output gaps.txt] {WRONLY CREAT EXCL}]
    set gaps 0
    # Avoid asking Vivado to materialize routes for the entire hierarchy.
    # The global route report still covers all data nets; inspect PARTIAL nets
    # explicitly and keep their complete routes to prove shared clock context.
    set partial_nets {}
    foreach net $all_nets status [get_property ROUTE_STATUS $all_nets] {
        dict incr counts $status
        if {$status eq "PARTIAL"} {lappend partial_nets $net}
    }
    puts "NTRPU_GEMV_QUAD_AUDIT_NETS total=[llength $all_nets] partial=[llength $partial_nets]"
    foreach net $partial_nets {
        set status [get_property ROUTE_STATUS $net]
        set route [get_property ROUTE $net]
        if {[string match -nocase *gap* $route]} {
            incr gaps
            puts $stream "NET $net STATUS $status\n$route"
        }
    }
    puts $stream "ROUTE_STRING_GAP_NETS $gaps\nROUTE_STATUS_COUNTS $counts\nROUTE_QUERY_SCOPE PARTIAL_NETS_ONLY"
    close $stream
    set stream [open [file join $output port_context.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\tdirection\tHD.CLK_SRC\tHD.PARTPIN_LOCS"
    foreach p [get_ports] {
        puts $stream "$p\t[get_property DIRECTION $p]\t[get_property HD.CLK_SRC $p]\t[get_property HD.PARTPIN_LOCS $p]"
    }
    close $stream
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    puts "NTRPU_GEMV_ROUTE_AUDIT_EXPORTED; read-only, no constraints or checkpoint modified"
    close_project
}
if {[catch {ntrpu_gemv_quad_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
