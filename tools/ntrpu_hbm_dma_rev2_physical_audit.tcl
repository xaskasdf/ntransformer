# Reopen an archived routed checkpoint; no optimization or routing commands.
set root [file normalize [file join [file dirname [info script]] ..]]
proc audit {root out} {
    file mkdir $out
    set_param general.maxThreads 2
    set f [open $root/tools/ntrpu_hbm_dma_rev2_physical.tcl];set text [read $f];close $f
    set boundary [string first "\nif \{\[catch \{physical" $text]
    if {$boundary<0} {error "Physical helper boundary changed"}
    uplevel #0 [string range $text 0 [expr {$boundary-1}]]
    open_checkpoint $root/evidence/routed.dcp
    if {[llength [get_clocks]]!=2 || [llength [get_cells -hier -filter {REF_NAME =~ BUFG*}]]!=2} {error "Unexpected clock inventory"}
    set cold [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]
    if {[llength $cold]!=10} {error "Changed cold-reset endpoints"}
    foreach p $cold {
        if {[get_property REF_PIN_NAME $p] ne "CLR" || ![get_property ASYNC_REG [get_cells -of_objects $p]]} {error "Changed cold-reset consumer"}
    }
    set f [open $out/cold_reset_endpoints.txt w];puts $f [join [lsort $cold] \n];close $f
    set f [open $out/design.tsv w]
    puts $f "PART\t[get_property PART [current_design]]"
    foreach flag {PLACED_FULLY PARTIALLY_ROUTED ROUTED_FULLY ERRORS_IN_ROUTES} {puts $f "$flag\t[get_property $flag [current_design]]"}
    puts $f "async_registers\t[llength [get_cells -hier -filter {ASYNC_REG == 1}]]"
    puts $f "blackboxes\t[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]"
    puts $f "latches\t[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]"
    close $f
    set f [open $out/clock_buffers.tsv w]
    puts $f "instance\tref\tloc\tclock\tperiod\toutput_net\troute_status\tloads\tinput_port\tce_driver"
    foreach inst {ntrpu_phy_reader_bufg ntrpu_phy_dma_bufg} port {clk dma_clk} name {reader_clk dma_clk} period {5.0 4.0} {
        set c [exact_cell $inst]
        set op [get_pins $inst/O];set nets [get_nets -segments -of_objects $op]
        set clock [get_clocks -of_objects $op]
        if {$clock ne $name || [get_property PERIOD $clock]!=$period} {error "Changed clock profile"}
        set loads [get_pins -leaf -of_objects $nets -filter {DIRECTION == IN}]
        if {![llength $loads]} {error "Unloaded clock buffer"}
        foreach load $loads {
            if {[get_property REF_PIN_NAME $load] ni {C CLK}} {error "Clock reaches a data/control pin: $load"}
        }
        set ins [get_nets -segments -of_objects [get_pins $inst/I]]
        set ip [get_ports -of_objects $ins]
        set il [get_pins -leaf -of_objects $ins -filter {DIRECTION == IN}]
        if {$ip ne $port || $il ne "$inst/I" || [llength [get_pins -quiet -leaf -of_objects $ins -filter {DIRECTION == OUT}]]} {error "Clock input bypass/source changed"}
        if {[get_property HD.CLK_SRC [get_ports $port]] ne ""} {error "Unexpected synthetic clock-source location"}
        set ce [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $inst/CE]] -filter {DIRECTION == OUT}]]
        if {[llength $ce]!=1 || [get_property REF_NAME $ce] ne "VCC"} {error "Clock enable is not static high"}
        puts $f "$inst\t[get_property REF_NAME $c]\t[get_property LOC $c]\t$clock\t$period\t[get_nets -of_objects $op]\t[get_property ROUTE_STATUS [get_nets -of_objects $op]]\t[llength $loads]\t$ip\t[get_property REF_NAME $ce]"
    }
    close $f
    set f [open $out/timing_subsets.tsv w]
    puts $f "subset\tkind\tslack\tstartpoint\tendpoint"
    foreach clock {reader_clk dma_clk} {
        set regs [all_registers -clock $clock]
        foreach {kind delay} {setup max hold min} {
            set p [get_timing_paths -from $regs -to $regs -delay_type $delay -max_paths 1]
            if {[llength $p]!=1} {error "Missing internal timing"}
            puts $f "${clock}_internal\t$kind\t[get_property SLACK $p]\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]"
            report_timing -from $regs -to $regs -delay_type $delay -max_paths 10 -input_pins -file $out/${clock}_internal_${kind}.rpt
            set p [get_timing_paths -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 1]
            puts $f "${clock}_all\t$kind\t[get_property SLACK $p]\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]"
        }
    }
    close $f
    topology_and_paths $root $out
    report_route_status -file $out/route_status.rpt
    report_timing_summary -report_unconstrained -max_paths 10 -file $out/timing.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored_exceptions.rpt
    report_bus_skew -max_paths 3 -file $out/bus_skew.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_drc -file $out/drc.rpt
    write_xdc -constraints all $out/effective_constraints.xdc
    puts NTRPU_BRIDGE_REV2_PHYSICAL_AUDIT_EXPORTED
}
if {[catch {audit $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
