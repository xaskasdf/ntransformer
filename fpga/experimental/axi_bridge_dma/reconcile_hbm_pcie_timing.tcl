# Query every original TIMING-16 PCIe endpoint without changing exceptions.
proc hp_reconcile {source out} {
    file mkdir $out
    set input [open $source/pcie_methodology_targets.tsv r]
    gets $input header
    set f [open $out/pcie_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "finding\tfrom\tto\tbaseline_warning_ns\tslack_ns\trequirement_ns\texception\tclock\tperiod_ns"
    while {[gets $input line]>=0} {
        lassign [split $line "\t"] finding from to warning
        set src [get_pins $from];set dst [get_pins $to]
        if {[llength $src]!=1 || [llength $dst]!=1} {error "PCIe methodology endpoint changed: $finding"}
        set paths [get_timing_paths -from $src -to $dst -delay_type max -max_paths 1]
        if {[llength $paths]!=1} {error "Missing PCIe methodology timing: $finding"}
        set clock [get_property STARTPOINT_CLOCK $paths]
        puts $f "$finding\t$from\t$to\t$warning\t[get_property SLACK $paths]\t[get_property REQUIREMENT $paths]\t[get_property EXCEPTION $paths]\t$clock\t[get_property PERIOD [get_clocks $clock]]"
        report_timing -from $src -to $dst -delay_type max -max_paths 1 -path_type full_clock_expanded -input_pins -file $out/${finding}.rpt
    }
    close $input;close $f
}
