# Read-only structural and path audit, also usable after reopening the DCP.
proc hc_audit {out} {
    file mkdir $out
    set f [open $out/paths.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tgroup\tfrom\tto\tstart_clock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach group [hc_groups] {
        lassign $group kind name from to budget skew count
        set paths [get_timing_paths -quiet -from $from -to $to -delay_type max -max_paths $count -nworst 1]
        if {[llength $paths]!=$count} {error "Incomplete paths for $name: [llength $paths]/$count"}
        set endpoints {}
        foreach path $paths {
            set endpoint [get_property ENDPOINT_PIN $path];lappend endpoints $endpoint
            set req [get_property REQUIREMENT $path];set slack [get_property SLACK $path]
            if {abs($req-$budget)>0.001} {error "Missing/overridden max delay $name: $req"}
            puts $f [join [list $kind $name [get_property STARTPOINT_PIN $path] $endpoint [get_property STARTPOINT_CLOCK $path] [get_property ENDPOINT_CLOCK $path] $req $slack [get_property DATAPATH_DELAY $path]] "\t"]
        }
        if {[lsort -unique $endpoints] ne [lsort -unique $to]} {error "Endpoint coverage mismatch $name"}
        report_timing -from $from -to $to -delay_type max -max_paths 1 -path_type full -file $out/${name}_worst.rpt
    }
    close $f
    set f [open $out/interstage.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tslack_ns\tclock\tsource_loc\tdest_loc"
    set pairs {}
    for {set lane 0} {$lane<2} {incr lane} {
        foreach fifo {commands returns} {
            foreach {a b} {ws1 ws2 rs1 rs2} {
                for {set bit 0} {$bit<3} {incr bit} {
                    lappend pairs [list [format {lanes[%d].%s/%s_reg[%d]} $lane $fifo $a $bit] [format {lanes[%d].%s/%s_reg[%d]} $lane $fifo $b $bit]]
                }
            }
        }
    }
    foreach chain {echo_r echo_m started_m failed_m armed_p failed_p stop_r failed_r} {
        for {set bit 0} {$bit<2} {incr bit} {
            lappend pairs [list [format {%s_reg[%d]} $chain $bit] [format {%s_reg[%d]} $chain [expr {$bit+1}]]]
        }
    }
    set sync [get_cells -hier -quiet -filter {ASYNC_REG == TRUE}]
    if {[llength $sync]!=72} {error "Expected 72 ASYNC_REG cells; got [llength $sync]"}
    foreach pair $pairs {
        lassign $pair a b
        set from [hc_pin $a/C];set to [hc_pin $b/D]
        foreach cell [list [get_cells -of_objects $from] [get_cells -of_objects $to]] {
            if {![get_property ASYNC_REG $cell]} {error "Synchronizer attribute missing $cell"}
        }
        set driver [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}]
        if {$driver ne "$a/Q"} {error "Logic inserted between synchronizer stages $a $b"}
        foreach type {max min} {
            set paths [get_timing_paths -quiet -from $from -to $to -delay_type $type -max_paths 1]
            if {[llength $paths]!=1} {error "Missing interstage timing $a $b $type"}
            set sc [get_property STARTPOINT_CLOCK $paths];set dc [get_property ENDPOINT_CLOCK $paths]
            if {$sc ne $dc} {error "Interstage clock mismatch"}
            puts $f [join [list $a $b $type [get_property SLACK $paths] $sc [get_property LOC [get_cells -of_objects $from]] [get_property LOC [get_cells -of_objects $to]]] "\t"]
        }
    }
    close $f
    set f [open $out/reset_endpoints.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tclock\treset_pin\treset_startpoints"
    foreach cell [get_cells -hier -filter {REF_NAME =~ FD*}] {
        set clock [get_clocks -of_objects [get_pins -of_objects $cell -filter {IS_CLOCK == 1}]]
        foreach pin [get_pins -of_objects $cell -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}] {
            set start [all_fanin -flat -startpoints_only -to $pin]
            set expected [dict get {pcie_clk p_cold_rst_n reader_clk r_cold_rst_n mgmt_clk m_cold_rst_n} $clock]
            if {$start ne $expected} {error "Reset contract mismatch $pin -> $start, expected $expected"}
            puts $f "$cell\t$clock\t$pin\t$start"
        }
    }
    close $f
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_cdc -details -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_route_status -file $out/route_status.rpt
    report_drc -file $out/drc.rpt
    report_methodology -file $out/methodology.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CDC_PHYSICAL_AUDIT_EXPORTED_PENDING_REVIEW;close $f
}
