# Reopen the frozen global-synthesis checkpoint; do not place without board LOCs.
set_param general.maxThreads 2
proc pin {name} {
    set p [get_pins -hierarchical -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin: $name"}
    return $p
}
proc stage_paths {out} {
    set f [open $out {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tslack\tclock"
    set chains {cold_m cold_a echo_a echo_m armed_a stop_a}
    for {set n 0} {$n<4} {incr n} {
        lappend chains [format {availability[%d].release_m} $n] [format {availability[%d].release_a} $n]
    }
    foreach chain $chains {
        for {set i 0} {$i<2} {incr i} {
            set from [pin [format {guarded/guard/%s_reg[%d]/C} $chain $i]]
            set to [pin [format {guarded/guard/%s_reg[%d]/D} $chain [expr {$i+1}]]]
            foreach kind {max min} {
                set p [get_timing_paths -quiet -no_report_unconstrained -from $from -to $to -delay_type $kind -max_paths 1]
                if {[llength $p]!=1} {error "Untimed synchronizer stage: $from -> $to $kind"}
                puts $f [join [list $from $to $kind [get_property SLACK $p] [get_property STARTPOINT_CLOCK $p]] "\t"]
            }
        }
    }
    close $f
}
if {[catch {
    if {[llength $argv]!=2} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv checkpoint out
    if {[file exists $out]} {error "New output required"}
    file mkdir $out
    open_checkpoint $checkpoint
    stage_paths $out/stages_before.tsv
    write_xdc -constraints all $out/before.xdc
    set f [open $out/native_cuts.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\tsource_ref\tsource_clock\tdest_clock\tbefore_paths\tafter_timed\tafter_ignored"
    foreach {n reg} {0 user_lnk_up_reg 1 user_reset_reg 2 user_reset_reg} {
        set from [pin ntrpu_bridge_i/inst/pcie4c_ip_i/inst/$reg/C]
        set source_clock [get_clocks -of_objects $from]
        if {[llength $source_clock]!=1 || [get_property PERIOD $source_clock]!=4.000} {error "Wrong native source clock"}
        for {set i 0} {$i<3} {incr i} {
            set to [pin [format {guarded/guard/availability[%d].release_m_reg[%d]/CLR} $n $i]]
            set starts [all_fanin -flat -startpoints_only -to $to]
            if {[lsort $starts] ne [lsort $from]} {error "Native source cone changed: $to $starts"}
            set dest [get_cells -of_objects $to]
            if {![get_property ASYNC_REG $dest]} {error "Destination is not ASYNC_REG"}
            set dc [get_clocks -of_objects [pin $dest/C]]
            if {$dc ne "mgmt_clk"} {error "Wrong destination clock"}
            set paths [get_timing_paths -quiet -no_report_unconstrained -from $from -to $to -max_paths 1]
            if {[llength $paths]!=1} {error "Expected timed native reset before the specific cut"}
            set_false_path -from $from -to $to
            set timed [get_timing_paths -quiet -no_report_unconstrained -from $from -to $to -max_paths 1]
            set ignored [get_timing_paths -quiet -user_ignored -from $from -to $to -max_paths 1]
            puts $f [join [list $from $to [get_property REF_NAME [get_cells -of_objects $from]] $source_clock $dc [llength $paths] [llength $timed] [llength $ignored]] "\t"]
            if {[llength $timed]!=0 || [llength $ignored]!=1} {error "Specific native cut not confirmed"}
        }
    }
    close $f
    stage_paths $out/stages_after.tsv
    # Starting at the primitive driver includes internal consumers even if the
    # diagnostic hierarchy output pin itself only leads to a top-level port.
    set f [open $out/control_consumers.tsv {WRONLY CREAT EXCL}]
    puts $f "signal\tdriver\tendpoint\tendpoint_clock"
    foreach signal {client_cold_rst_n client_link_up client_stop} {
        set nets [get_nets -segments -of_objects [pin guarded/$signal]]
        set driver [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
        if {[llength $driver]!=1} {error "Nonunique control driver"}
        set count 0
        foreach endpoint [lsort [all_fanout -flat -endpoints_only -from $driver]] {
            if {![string match guarded/client/* $endpoint]} {continue}
            incr count
            set clocks [get_clocks -quiet -of_objects [get_pins -of_objects [get_cells -of_objects $endpoint] -filter {IS_CLOCK == 1}]]
            puts $f [join [list $signal $driver $endpoint [join $clocks ,]] "\t"]
        }
        if {$count==0} {error "Control no longer reaches client: $signal"}
    }
    close $f
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_methodology -file $out/methodology.rpt
    report_drc -file $out/drc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_timing -from [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && IS_PRIMITIVE == 1}] -to [get_cells -hierarchical -filter {NAME =~ guarded/client/* && IS_PRIMITIVE == 1}] -delay_type min_max -max_paths 20 -file $out/guard_to_client.rpt
    write_xdc -constraints all $out/after.xdc
    write_checkpoint $out/native_audited.dcp
    set f [open $out/physical_gate.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    puts $f "refclk_ibuf\tLOC\t[get_property LOC [get_cells refclk_ibuf]]"
    foreach p {sys_clk_p sys_clk_n sys_rst_n mgmt_clk_in} {
        puts $f "$p\tPACKAGE_PIN\t[get_property PACKAGE_PIN [get_ports $p]]"
        puts $f "$p\tIOSTANDARD\t[get_property IOSTANDARD [get_ports $p]]"
    }
    puts $f "placement\texecuted\tfalse"
    puts $f "routing\texecuted\tfalse"
    close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f "NATIVE_BOUNDARY_AUDITED_BOARD_CONTRACT_PENDING"
    close $f
    close_design
} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
