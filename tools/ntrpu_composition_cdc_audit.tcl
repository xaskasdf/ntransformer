# Read-only checkpoint audit; does not rerun synthesis, placement or routing.
set script_dir [file dirname [file normalize [info script]]]
proc cdc_audit {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint [file normalize [file join $script_dir .. evidence routed.dcp]]
    report_bus_skew -delay_type min_max -max_paths 1 -warn_on_violation -file [file join $output bus_skew.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc_unwaived.rpt]
    help report_waivers
    help get_waivers
    set f [open [file join $output clock_binding.txt] w]
    set converters [get_cells -hier -filter {REF_NAME =~ ntrpu_peer_memory_cdc*}]
    puts $f "CONVERTERS=[llength $converters]"
    if {[llength $converters]!=2} {error "Expected two CDC instances"}
    foreach cell $converters {
        foreach port {s_axi_aclk m_axi_aclk} {
            set pin [get_pins -of_objects $cell -filter "REF_PIN_NAME == $port"]
            set clocks [get_clocks -of_objects $pin]
            if {[llength $pin]!=1 || [llength $clocks]!=1} {error "Ambiguous clock on $cell/$port"}
            puts $f "$cell/$port=$clocks"
        }
    }
    foreach cell [get_cells -hier -filter {REF_NAME==BUFGCE}] {
        set net [get_nets -of_objects [get_pins $cell/O]]
        puts $f "BUFG=$cell CLOCK=[get_clocks -of_objects $net] ROUTE=[get_property ROUTE_STATUS $net]"
    }
    close $f
    close_design
    puts NTRPU_COMPOSITION_CDC_AUDIT_EXPORTED
}
if {[catch {cdc_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
