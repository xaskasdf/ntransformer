# Read-only audit: reconcile methodology diagnostics with final routed timing.
proc ntrpu_timing_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected ROUTED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_ports]] != 221} { error "Unexpected joint checkpoint" }
    foreach cmd {report_methodology get_methodology_violations report_timing get_timing_paths report_cdc} {
        help $cmd
    }
    report_methodology -file [file join $output methodology_before.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    set from [get_pins {u_pcie/ntrpu_xdma_i/inst/pcie4_ip_i/inst/ntrpu_xdma_pcie4_ip_pcie_4_0_pipe_inst/pcie_4_0_e4_inst/CORECLK}]
    set to [get_pins {u_pcie/ntrpu_xdma_i/inst/udma_wrapper/dma_top/base/IRQ_INST/clk_msi_req_reg[0]/D}]
    if {[llength $from] != 1 || [llength $to] != 1} { error "Diagnostic endpoints not unique" }
    report_timing -from $from -to $to -delay_type max -path_type full_clock_expanded -max_paths 16 -file [file join $output named_path.rpt]
    set paths [get_timing_paths -from $from -to $to -delay_type max -max_paths 16]
    set stream [open [file join $output named_paths.txt] {WRONLY CREAT EXCL}]
    puts $stream "path_count=[llength $paths]"
    foreach path $paths { puts $stream [list [get_property SLACK $path] [get_property STARTPOINT_PIN $path] [get_property ENDPOINT_PIN $path]] }
    close $stream
    report_timing -delay_type max -max_paths 10 -path_type full_clock_expanded -file [file join $output worst_paths.rpt]
    report_methodology -file [file join $output methodology_after.rpt]
    set violations [get_methodology_violations -quiet]
    set i 0
    foreach violation $violations {
        report_property -all -file [file join $output violation_[incr i].rpt] $violation
    }
    report_clocks -file [file join $output clocks.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -file [file join $output cdc_details.rpt]
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_route_status -file [file join $output route.rpt]
    puts "NTRPU_INTERFACE_TIMING_AUDIT_EXPORTED; read-only diagnostics, no waivers or design changes"
    close_project
}
if {[catch {ntrpu_timing_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
