# Additional topology proof on the reopened, already routed checkpoint.
proc hc_topology {out} {
    set f [open $out/topology.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tfrom\tto\tdriver_ref\tsource_loc\tdest_loc\tleaf_loads"
    foreach group [hc_groups] {
        lassign $group kind name sources destinations budget skew count
        foreach to $destinations {
            set drivers [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}]
            if {[llength $drivers]!=1} {error "Not a single primitive driver: $to"}
            set driver [get_cells -of_objects $drivers]
            if {[lsearch -exact $sources $driver]<0} {error "Unexpected logic before CDC endpoint: $to"}
            set dest [get_cells -of_objects $to]
            set loads [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $drivers] -filter {DIRECTION == IN}]
            if {$kind eq "capture"} {
                if {[get_property REF_NAME $dest] ne "FDRE" || $loads ne $to} {error "RAM bit bypasses its capture register: $to"}
            } else {
                if {![get_property ASYNC_REG $dest]} {error "CDC first stage lacks ASYNC_REG: $to"}
            }
            puts $f [join [list $kind $drivers $to [get_property REF_NAME $driver] [get_property LOC $driver] [get_property LOC $dest] [llength $loads]] "\t"]
        }
    }
    close $f
    set f [open $out/first_stage_loads.tsv {WRONLY CREAT EXCL}]
    puts $f "source\tdestination\tsource_loc\tdest_loc"
    set count 0
    foreach cell [get_cells -quiet -hier -filter {ASYNC_REG == TRUE}] {
        set q [get_pins -of_objects $cell -filter {REF_PIN_NAME == Q}]
        set loads [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $q] -filter {DIRECTION == IN}]
        set sync_loads {}
        foreach load $loads {
            set dest [get_cells -of_objects $load]
            if {[get_property ASYNC_REG $dest] in {1 TRUE true} && [get_property REF_PIN_NAME $load] eq "D"} {
                set sc [get_clocks -of_objects [get_pins -of_objects $cell -filter {IS_CLOCK == 1}]]
                set dc [get_clocks -of_objects [get_pins -of_objects $dest -filter {IS_CLOCK == 1}]]
                if {$sc eq $dc} {lappend sync_loads $load}
            }
        }
        if {[llength $sync_loads]} {
            if {[llength $loads]!=1} {error "Synchronizer intermediate stage has additional consumers: $cell"}
            set dest [get_cells -of_objects $sync_loads]
            puts $f "$cell\t$dest\t[get_property LOC $cell]\t[get_property LOC $dest]"
            incr count
        }
    }
    if {$count!=40} {error "Unexpected synchronizer stage connections: $count"}
    close $f
    set f [open $out/parent_clocks.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tclock\tperiod_ns\tassumed_bufg\tnet\troute_status\tleaf_drivers"
    foreach {port clock period site} {p_clk pcie_clk 4.0 BUFGCE_X0Y60 r_clk reader_clk 5.0 BUFGCE_X0Y61 mgmt_clk mgmt_clk 10.0 BUFGCE_X0Y62} {
        set object [get_ports $port]
        if {[get_property HD.CLK_SRC $object] ne $site || [get_property PERIOD [get_clocks $clock]]!=$period} {error "Parent clock context changed"}
        set net [get_nets -of_objects $object]
        set drivers [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $object] -filter {DIRECTION == OUT}]
        puts $f "$port\t$clock\t$period\t$site\t$net\t[get_property ROUTE_STATUS $net]\t[llength $drivers]"
        report_property -all $net -file $out/${port}_net.rpt
    }
    close $f
    set gaps [report_route_status -return_nets -route_type GAPS]
    set f [open $out/clock_gaps.tsv {WRONLY CREAT EXCL}]
    puts $f "net\tports"
    foreach net $gaps {
        puts $f "$net\t[join [get_ports -of_objects [get_nets -segments $net]] ,]"
    }
    close $f
    if {[lsort $gaps] ne [lsort [get_nets -of_objects [get_ports {p_clk r_clk mgmt_clk}]]]} {error "Explicit gaps are not exactly the three parent clocks"}
    report_drc -checks [get_drc_checks RTSTAT-10] -file $out/boundary_drc.rpt
    set violations [get_drc_violations -quiet *RTSTAT-10*]
    if {[llength $violations]!=1} {error "Changed boundary DRC inventory"}
    set f [open $out/boundary_nets.tsv {WRONLY CREAT EXCL}]
    puts $f "net\tleaf_loads\tports"
    foreach net [get_nets -of_objects $violations] {
        set segments [get_nets -segments $net]
        set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
        puts $f "$net\t[llength $loads]\t[join [get_ports -quiet -of_objects $segments] ,]"
    }
    close $f
}
