source [file join [file dirname [file normalize [info script]]] ntrpu_retained_start_reset_constraints.tcl]
proc retained_start_scoped_timing {output checkpoint mode} {
    if {$mode ne "start" || [file exists $output]} {error "Expected new output and start mode"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    report_methodology -file [file join $output before_methodology.rpt]
    report_timing_summary -file [file join $output before_timing.rpt]
    set before [dict create]
    foreach cell [get_cells -hierarchical -filter {REF_NAME==FDCE}] {
        if {![regexp {/release_ff_reg\[[12]\]$} $cell]} {continue}
        set dp [get_pins -of $cell -filter {REF_PIN_NAME==D}]
        set path [get_timing_paths -quiet -no_report_unconstrained -to $dp -max_paths 1]
        if {[llength $path]!=1} {error "Missing baseline stage-to-stage timing $dp"}
        dict set before $dp [list [get_property SLACK $path] [get_property STARTPOINT_PIN $path] [get_property ENDPOINT_PIN $path]]
    }
    if {[dict size $before]!=30} {error "Expected thirty timed synchronizer D paths"}
    set inventory [ntrpu_retained_start_reset_constraints {}]
    set f [open [file join $output scoped_reset_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin"
    foreach pin [dict get $inventory clear_pins] {puts $f $pin}
    close $f
    set f [open [file join $output data_path_comparison.tsv] {WRONLY CREAT EXCL}]
    puts $f "endpoint\tbefore_slack\tafter_slack\tbefore_startpoint\tafter_startpoint"
    foreach pair [dict get $inventory data_paths] {
        lassign $pair previous dp
        set path [get_timing_paths -quiet -no_report_unconstrained -from $previous -to $dp -max_paths 1]
        if {[llength $path]!=1} {error "Scoped constraint removed D path $dp"}
        set values [list [get_property SLACK $path] [get_property STARTPOINT_PIN $path] [get_property ENDPOINT_PIN $path]]
        if {$values ne [dict get $before $dp]} {error "Stage-to-stage timing changed $dp"}
        puts $f "$dp\t[lindex $values 0]\t[lindex $values 0]\t[lindex $values 1]\t[lindex $values 1]"
    }
    close $f
    set remaining 0
    foreach source {mgmt_clk source_clk ui0_clk ui1_clk} {
        foreach dest {mgmt_clk source_clk ui0_clk ui1_clk} {
            if {$source eq $dest} {continue}
            incr remaining [llength [get_timing_paths -quiet -no_report_unconstrained -from [get_clocks $source] -to [get_clocks $dest] -max_paths 1000]]
        }
    }
    if {$remaining!=0} {error "Unresolved cross-clock timed paths $remaining"}
    report_methodology -file [file join $output after_methodology.rpt]
    report_timing_summary -file [file join $output after_timing.rpt]
    check_timing -verbose -file [file join $output after_check_timing.rpt]
    report_cdc -details -file [file join $output after_cdc.rpt]
    report_exceptions -coverage -file [file join $output after_exceptions.rpt]
    report_drc -file [file join $output after_drc.rpt]
    report_waivers -file [file join $output after_waivers.rpt]
    write_xdc -constraints all [file join $output effective_scoped.xdc]
    # No DCP is written: this is a bounded constraint trial, pending review.
    close_design
    puts NTRPU_RETAINED_START_SCOPED_TIMING_EXPORTED
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_start_scoped_timing [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]] [lindex $argv 2]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
