set root [file normalize [file join [file dirname [info script]] ..]]
proc review_constraints {root out checkpoint} {
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    set cells [get_cells -quiet -hier -regexp {(^|.*/)startup/(ready_meta_reg\[[0-2]\]|fault_meta_reg\[[0-1]\])}]
    if {[llength $cells]!=5} {error "Expected five startup status synchronizers"}
    foreach c $cells {if {![get_property ASYNC_REG $c]} {error "Missing ASYNC_REG on $c"}}
    foreach clock {memory_clk transport_clk} {if {[llength [get_clocks $clock]]!=1} {error "Missing $clock"}}
    read_xdc [file join $root fpga constraints ntrpu_hbm_startup_cdc.xdc]
    if {[llength $ntrpu_status_internal_d]!=3 || [llength $ntrpu_status_apb_d]!=2} {error "Unexpected first-stage pins"}
    set f [open [file join $out status_endpoints.tsv] w]
    foreach c [lsort [get_cells -hier -regexp {(^|.*/)startup/(ready_meta_reg\[[0-2]\]|fault_meta_reg\[[0-1]\])}]] {
        set dpin [get_pins -of_objects $c -filter {REF_PIN_NAME == D}]
        puts $f "$c\t[get_property ASYNC_REG $c]\t$dpin\t[get_property PERIOD [get_clocks -of_objects [get_pins -of_objects $c -filter {REF_PIN_NAME == C}]]]"
    };close $f
    report_clock_interaction -file [file join $out clock_interaction.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $out cdc.rpt]
    report_timing -from [get_ports init_complete_async*] -max_paths 10 -file [file join $out external_status.rpt]
    report_timing -from [get_clocks memory_clk] -to [get_clocks transport_clk] -max_paths 10 -file [file join $out status_crossing.rpt]
    report_exceptions -coverage -file [file join $out exceptions.rpt]
    check_timing -verbose -file [file join $out check_timing.rpt]
    report_drc -file [file join $out drc.rpt]
    set f [open [file join $out metrics.txt] w]
    foreach c {transport_clk memory_clk} {
        foreach {kind delay} {setup max hold min} {
            puts $f "${kind}_$c [get_property SLACK [get_timing_paths -from [get_clocks $c] -to [get_clocks $c] -delay_type $delay -max_paths 1]]"
        }
    };close $f
    write_checkpoint [file join $out async_constrained.dcp]
    write_xdc -constraints all [file join $out constraints.xdc]
    close_design
    puts NTRPU_HBM_CDC_CONSTRAINTS_COMPLETE
}
if {[llength $argv]!=2} {error "Expected output and checkpoint"}
if {[catch {review_constraints $root [lindex $argv 0] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
