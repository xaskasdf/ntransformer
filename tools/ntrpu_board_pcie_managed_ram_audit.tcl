# Read-only bounded audit of the sixteen SYNTH-6 pool RAM instances.
# Preserve clocks, reset exceptions, RTL and all concurrent validations.
proc audit_managed_ram {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected CONSTRAINED_DCP NEW_OUTPUT"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_dual_host_cold" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_clocks]] != 5} {error "Unexpected checkpoint"}
    set cells [lsort [get_cells -hierarchical -filter {REF_NAME =~ RAMB*}]]
    set expected {}
    for {set reader 0} {$reader < 2} {incr reader} {
        for {set slice 0} {$slice < 8} {incr slice} {
            lappend expected [format {core/receiver/receiver/receiver/core/packed_pool.reader[%d].payload_reg_%d} $reader $slice]
        }
    }
    if {$cells ne [lsort $expected]} {error "RAM cell inventory changed"}
    set properties {REF_NAME DOA_REG DOB_REG READ_WIDTH_A READ_WIDTH_B WRITE_WIDTH_A WRITE_WIDTH_B WRITE_MODE_A WRITE_MODE_B}
    set summary [open [file join $output ram.tsv] {WRONLY CREAT EXCL}]
    puts $summary "cell\t[join $properties \t]"
    set pins [open [file join $output ram_pins.tsv] {WRONLY CREAT EXCL}]
    puts $pins "pin\tdirection\tnets\tclocks"
    foreach cell $cells {
        set values {}
        foreach property $properties {lappend values [get_property $property $cell]}
        puts $summary "$cell\t[join $values \t]"
        foreach pin [lsort [get_pins -of_objects $cell]] {
            set nets [get_nets -quiet -of_objects $pin]
            set clocks [get_clocks -quiet -of_objects $pin]
            puts $pins "$pin\t[get_property DIRECTION $pin]\t[join $nets ,]\t[join $clocks ,]"
        }
    }
    close $summary
    close $pins
    set outputs [get_pins -of_objects $cells -filter {DIRECTION == OUT}]
    foreach delay {max min} {
        report_timing -from $outputs -delay_type $delay -max_paths 20 -nworst 1 -file [file join $output ram_${delay}_timing.rpt]
    }
    set counters [lsort [get_cells -quiet -hierarchical -filter {NAME =~ *remaining_reg* && IS_PRIMITIVE == 1}]]
    set f [open [file join $output counter_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name"
    foreach cell $counters {puts $f "$cell\t[get_property REF_NAME $cell]"}
    close $f
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_drc -file [file join $output drc.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    puts "NTRPU_MANAGED_RAM_AUDIT_EXPORTED cells=[llength $cells] counter_cells=[llength $counters]"
    close_project
}
if {[catch {audit_managed_ram $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
