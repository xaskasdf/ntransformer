# Read-only diagnostic of the archived, single-channel generic DDR design.
# R2 uses TOP for the module identity and records the distinct design NAME.
proc ntrpu_power_probe {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected a fresh Vivado process: INPUT_DCP NEW_OUTPUT_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set checkpoint [file normalize $checkpoint]
    set output [file normalize $output]
    if {![file isfile $checkpoint]} { error "Input checkpoint is missing" }
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    set identity [open [file join $output design_identity.tsv] {WRONLY CREAT EXCL}]
    foreach property {NAME TOP PART} {
        puts $identity "$property\t[get_property $property [current_design]]"
    }
    puts $identity "PORTS\t[llength [get_ports]]"
    close $identity
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property TOP [current_design]] ne "example_top" ||
        [llength [get_ports]] != 154} {
        error "Unexpected design; require the manifest checkpoint"
    }
    set stream [open [file join $output scope.txt] {WRONLY CREAT EXCL}]
    puts $stream "Scope: one generic DDR4 channel, two slots, not qualified F1"
    puts $stream "No new activity or thermal constraints applied"
    puts $stream "Inherited activity plus tool defaults; diagnostic only"
    puts $stream "Not a four-channel board, PCIe or NTRPU power budget"
    puts $stream "Input: $checkpoint"
    puts $stream "Vivado: [version -short]"
    close $stream
    report_clocks -file [file join $output clocks.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_power -advisory -file [file join $output power.rpt]
    report_power -format xml -file [file join $output power.xml]
    close_project
    puts "NTRPU_POWER_DIAGNOSTIC_EXPORTED; NOT_POWER_QUALIFIED"
}
if {[catch {ntrpu_power_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
