# Audit the frozen asynchronous composition; no hard HBM instance, no waivers.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc audit {root output} {
    set_param general.maxThreads 2
    # Execute only the accepted project's IP and RTL setup, never its testbench.
    set text [read_text [file join $root tools ntrpu_hbm_cdc_async.tcl]]
    set begin [string first {    create_project hbm_adapter} $text]
    set end [string first {    add_files -fileset sim_1} $text]
    if {$begin<0 || $end<=$begin} {error "Frozen setup boundaries changed"}
    eval [string range $text $begin [expr {$end-1}]]
    set xdc [file join $output audit_clocks.xdc]
    set f [open $xdc w]
    puts $f {create_clock -name transport_clk -period 5.000 [get_ports clk]}
    puts $f {create_clock -name memory_clk -period 2.222 [get_ports mem_clk]}
    puts $f {create_clock -name apb_status_clk -period 10.000}
    puts $f {set_input_delay -clock apb_status_clk -min 0.000 [get_ports init_complete_async*]}
    puts $f {set_input_delay -clock apb_status_clk -max 0.000 [get_ports init_complete_async*]}
    puts $f {set_clock_uncertainty -setup 0.100 [get_clocks {transport_clk memory_clk}]}
    puts $f {set_clock_uncertainty -hold 0.050 [get_clocks {transport_clk memory_clk}]}
    close $f
    add_files -fileset constrs_1 $xdc
    synth_design -top ntrpu_hbm_transport_async -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
    write_checkpoint [file join $output async_synth.dcp]
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc_without_waivers.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_drc -file [file join $output drc.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    write_xdc -constraints all [file join $output constraints.xdc]
    set f [open [file join $output metrics.txt] w]
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    foreach c {transport_clk memory_clk} {
        set clock [get_clocks $c]
        puts $f "PERIOD_$c [get_property PERIOD $clock]"
        foreach {kind delay} {setup max hold min} {
            report_timing -from $clock -to $clock -delay_type $delay -max_paths 10 -file [file join $output ${c}_${kind}.rpt]
            puts $f "${kind}_$c [get_property SLACK [get_timing_paths -from $clock -to $clock -delay_type $delay -max_paths 1]]"
        }
    }
    close $f
    set f [open [file join $output async_registers.tsv] w]
    foreach c [lsort [get_cells -hier -filter {ASYNC_REG == TRUE}]] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property ASYNC_REG $c]"
    };close $f
    close_project
    puts NTRPU_HBM_CDC_AUDIT_COMPLETE
}
if {[llength $argv]!=1} {error "Expected output"}
if {[catch {audit $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
