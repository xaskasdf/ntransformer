# Reopen the exact R5 DCP and inspect exceptions without changing constraints.
# No resynthesis, clock relaxation, path exception, optimization or placement.
proc inventory_timing_recheck {input output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    file copy [file join $input pcie_request_inventory.dcp] [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PERIOD [get_clocks pcie_user_clk]]!=4.000} {error "Unexpected clock period"}
    set inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set outputs [get_ports -filter {DIRECTION == OUT}]
    if {[llength $inputs]!=1351 || [llength $outputs]!=412} {error "Unexpected inventory interface"}
    # Inspect the restored timing state before applying any new constraints.
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    report_exceptions -summary -file [file join $output exception_summary.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_utilization -file [file join $output utilization.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    set f [open [file join $output ports.tsv] w]
    puts $f "port\tdirection"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    set f [open [file join $output primitives.tsv] w]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {puts $f "$cell\t[get_property REF_NAME $cell]"}
    close $f
    write_xdc [file join $output complete_constraints.xdc]
    write_checkpoint [file join $output constrained_inventory.dcp]
    close_design
    puts NTRPU_PCIE_INVENTORY_TIMING_RECHECK_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_timing_recheck [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
