proc ct_audit {out routed} {
    file mkdir $out
    set f [open $out/paths.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tgroup\tfrom\tto\tstart_clock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach group [ct_groups] {
        lassign $group kind name from to budget skew count
        set paths [get_timing_paths -quiet -from $from -to $to -delay_type max -max_paths $count -nworst 1]
        if {[llength $paths]!=$count} {error "Incomplete constrained paths $name"}
        set endpoints {}
        foreach path $paths {
            set end [get_property ENDPOINT_PIN $path];lappend endpoints $end
            if {abs([get_property REQUIREMENT $path]-$budget)>0.001} {error "Constraint overridden $name"}
            puts $f [join [list $kind $name [get_property STARTPOINT_PIN $path] $end [get_property STARTPOINT_CLOCK $path] [get_property ENDPOINT_CLOCK $path] [get_property REQUIREMENT $path] [get_property SLACK $path] [get_property DATAPATH_DELAY $path]] "\t"]
        }
        if {[lsort -unique $endpoints] ne [lsort -unique $to]} {error "Endpoint set mismatch $name"}
    };close $f
    set f [open $out/clocks.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tclock\tperiod_ns\tbuffer\tsite\tclock_loads\tinput_startpoint\troute_status"
    foreach {port clock period} {clk pcie_clk 4.0 reader_clk reader_clk 5.0 mgmt_clk mgmt_clk 10.0 mem_clk mem_clk 2.222} {
        set buf [ct_one [get_cells clock_probe_$clock] "$clock buffer"]
        if {[get_property REF_NAME $buf] ne "BUFGCE"} {error "Missing clock primitive"}
        if {[get_property PERIOD [get_clocks $clock]]!=$period} {error "Changed period"}
        set net [ct_one [get_nets -of_objects [get_pins $buf/O]] "$buf output"]
        set loads [get_pins -leaf -of_objects [get_nets -segments $net] -filter {IS_CLOCK == 1}]
        if {[llength $loads]<1} {error "Empty clock domain $clock"}
        set start [all_fanin -flat -startpoints_only -to [get_pins $buf/I]]
        if {$start ne $port} {error "Unexpected clock source $clock => $start"}
        puts $f "$port\t$clock\t$period\t$buf\t[get_property LOC $buf]\t[llength $loads]\t$start\t[get_property ROUTE_STATUS $net]"
    };close $f
    report_cdc -details -file $out/cdc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_drc -file $out/drc.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    if {$routed} {
        foreach clock {pcie_clk reader_clk mgmt_clk mem_clk} {
            report_timing -from [get_clocks $clock] -to [get_clocks $clock] -delay_type max -max_paths 8 -nworst 1 -file $out/${clock}_setup.rpt
            report_timing -from [get_clocks $clock] -to [get_clocks $clock] -delay_type min -max_paths 4 -nworst 1 -file $out/${clock}_hold.rpt
        }
        report_route_status -file $out/route_status.rpt
        report_methodology -file $out/methodology.rpt
        set f [open $out/route_gaps.tsv {WRONLY CREAT EXCL}];puts $f "net\tstatus\tports\tleaf_drivers"
        foreach net [report_route_status -return_nets -route_type GAPS] {
            set segments [get_nets -segments $net]
            puts $f "$net\t[get_property ROUTE_STATUS $net]\t[join [get_ports -quiet -of_objects $segments] ,]\t[join [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}] ,]"
        };close $f
    }
    write_xdc -constraints all $out/effective.xdc
}
