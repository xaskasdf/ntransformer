# Exact bindings for the frozen VU47P transport, no clock-group/false-path cuts.
# 4 ns data bound; 2 ns Gray skew (half the fastest 4 ns period).
proc hc_cells {pattern expected} {
    set cells [get_cells -quiet -hier -regexp $pattern]
    if {[llength $cells]!=$expected} {error "Binding $pattern: expected $expected, got [llength $cells]"}
    return $cells
}
proc hc_pin {name} {
    set pins [get_pins -quiet -hier -filter [format {NAME == "%s"} $name]]
    if {[llength $pins]!=1} {error "Missing exact pin $name"}
    return $pins
}
proc hc_clock {cells expected} {
    set pins [get_pins -of_objects $cells -filter {IS_CLOCK == 1}]
    set clocks [lsort -unique [get_clocks -of_objects $pins]]
    if {$clocks ne $expected} {error "Clock mismatch: $cells => $clocks, expected $expected"}
}
proc hc_groups {} {
    set groups {}
    for {set lane 0} {$lane<2} {incr lane} {
        foreach {fifo width wclock rclock} {commands 136 pcie_clk reader_clk returns 593 reader_clk pcie_clk} {
            foreach {src dst sc dc} [list wg ws1 $wclock $rclock rg rs1 $rclock $wclock] {
                set from [hc_cells [format {^lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $src] 3]
                set to [hc_cells [format {^lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $dst] 3]
                hc_clock $from $sc;hc_clock $to $dc
                lappend groups [list gray l${lane}_${fifo}_${src} $from [get_pins -of_objects $to -filter {REF_PIN_NAME == D}] 4.0 2.0 3]
            }
            set output [hc_cells [format {^lanes\[%d\]\.%s/output_data_reg\[[0-9]+\]$} $lane $fifo] $width]
            hc_clock $output $rclock
            set to [get_pins -of_objects $output -filter {REF_PIN_NAME == D}]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}]
            set from [lsort -unique [get_cells -of_objects $drivers]]
            if {[llength $from]!=$width} {error "RAM driver coverage differs for $lane $fifo"}
            foreach ram $from {
                if {[get_property REF_NAME $ram] ni {RAMD32 RAMS32}} {error "Unexpected RAM driver $ram"}
            }
            hc_clock $from $wclock
            lappend groups [list capture l${lane}_${fifo}_data $from $to 4.0 0 $width]
        }
    }
    foreach {src dst sc dc} {
        request echo_r mgmt_clk reader_clk
        {echo_r_reg[2]} echo_m reader_clk mgmt_clk
        reader_failed failed_m reader_clk mgmt_clk
        armed armed_p mgmt_clk pcie_clk
        management_fault failed_p mgmt_clk pcie_clk
        management_fault failed_r mgmt_clk reader_clk
        stop_held stop_r pcie_clk reader_clk
    } {
        if {![string match *_reg* $src]} {append src _reg}
        set from [get_cells -quiet -hier -filter [format {NAME == "%s"} $src]]
        if {[llength $from]!=1} {error "Missing control source $src"}
        set to [hc_pin [format {%s_reg[0]/D} $dst]]
        hc_clock $from $sc;hc_clock [get_cells -of_objects $to] $dc
        lappend groups [list control $dst $from $to 4.0 0 1]
    }
    return $groups
}
proc hc_constrain {out} {
    set f [open $out/bindings.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tname\tsources\tendpoints\tmax_ns\tskew_ns"
    foreach group [hc_groups] {
        lassign $group kind name from to budget skew count
        puts $f "$kind\t$name\t[llength $from]\t[llength $to]\t$budget\t$skew"
        set_max_delay -datapath_only $budget -from $from -to $to
        if {$skew>0} {set_bus_skew $skew -from $from -to $to}
    }
    close $f
}
