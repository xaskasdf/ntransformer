# Read-only localization of all eleven V3 TIMING-16 endpoints.
set_param general.maxThreads 2
proc exact {name} {
    set p [get_pins -hierarchical -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin: $name"}
    return $p
}
if {[catch {
    if {[llength $argv]!=2} {error "ROUTED_CHECKPOINT NEW_OUTPUT required"}
    lassign $argv dcp out
    if {[file exists $out]} {error "New output required"};file mkdir $out
    open_checkpoint $dcp
    write_xdc -constraints all $out/before.xdc
    set f [open $out/reset_sources.tsv {WRONLY CREAT EXCL}]
    puts $f "destination\tstartpoints\tdrivers"
    for {set n 0} {$n<3} {incr n} {
        for {set i 0} {$i<3} {incr i} {
            set p [exact [format {guarded/guard/availability[%d].release_m_reg[%d]/CLR} $n $i]]
            set starts [all_fanin -flat -startpoints_only -to $p]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $p] -filter {DIRECTION == OUT}]
            puts $f [join [list $p [join $starts ,] [join $drivers ,]] "\t"]
        }
    }
    close $f
    set from [exact ntrpu_bridge_i/inst/pcie4c_ip_i/inst/ntrpu_bridge_pcie4c_ip_pcie_4_0_pipe_inst/pcie_4_c_e4_inst/CORECLK]
    set prefix ntrpu_bridge_i/inst/udma_wrapper/dma_top/axi4mm_bridge_top_inst/mm_slave_port.axi_enhanced_cfg/axi_enhanced_cfg_block_bridge_inst/end_point_1
    set f [open $out/methodology_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "endpoint\tmode\tcount\tslack\tstart_clock\tend_clock"
    foreach item {rdata_1 rdata_21 rdata_14 rdata_22 rdata_2 rdata_9 rdata_20 rdata_25 rdata_31 fsm_0 fsm_1} {
        lassign [split $item _] kind bit
        if {$kind eq "rdata"} {
            set to [exact [format {%s.s_axi_ctl_rdata_reg[%d]/CE} $prefix $bit]]
        } else {
            set to [exact [format {ntrpu_bridge_i/inst/udma_wrapper/dma_top/axi4mm_bridge_top_inst/mm_slave_port.axi4mm_axi_slave_read_inst/FSM_sequential_read_reqSM_cs_reg[%d]/CE} $bit]]
        }
        foreach mode {timed ignored} {
            set args [list -quiet -from $from -to $to -delay_type max -max_paths 1 -no_report_unconstrained]
            if {$mode eq "ignored"} {lappend args -user_ignored}
            set paths [get_timing_paths {*}$args]
            if {[llength $paths]} {
                puts $f "$to\t$mode\t[llength $paths]\t[get_property SLACK $paths]\t[get_property STARTPOINT_CLOCK $paths]\t[get_property ENDPOINT_CLOCK $paths]"
                report_timing -of_objects $paths -path_type full_clock_expanded -file $out/path_${item}_${mode}.rpt
            } else {puts $f "$to\t$mode\t0\t\t\t"}
        }
    }
    close $f
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_methodology -file $out/methodology.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f ROUTED_FINDINGS_READONLY_PROBE_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
