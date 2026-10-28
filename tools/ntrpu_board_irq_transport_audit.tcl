# Read-only inspection of initialization and protected bus constraints.
proc ntrpu_irq_transport_audit {arguments} {
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {error "Expected SYNTH_DCP NEW_DIRECTORY"}
    lassign $arguments input output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $input [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_irq_transport" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Checkpoint identity"}
    set registers [get_cells -quiet -hierarchical -filter {NAME =~ event_payload_reg* || NAME =~ service_payload_reg* || NAME == event_send_reg || NAME == service_send_reg}]
    if {![llength $registers]} {error "No transport storage found"}
    set f [open [file join $output initialization.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit"
    foreach cell [lsort $registers] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]"
    }
    close $f
    foreach cell {event_send_reg service_send_reg} {
        if {[llength [get_cells -quiet $cell]]!=1 || [get_property INIT [get_cells $cell]] ne "1'b0"} {error "Send configuration initialization"}
    }
    set f [open [file join $output waivers.txt] {WRONLY CREAT EXCL}]
    set waivers [get_waivers -quiet]
    foreach waiver $waivers {
        puts $f "$waiver"
        report_property -all $waiver -file [file join $output waiver-[string map {/ _ : _} $waiver].rpt]
    }
    close $f
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    puts "NTRPU_IRQ_AUDIT_EXPORTED registers=[llength $registers] waivers=[llength $waivers]"
    close_project
}
if {[catch {ntrpu_irq_transport_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
