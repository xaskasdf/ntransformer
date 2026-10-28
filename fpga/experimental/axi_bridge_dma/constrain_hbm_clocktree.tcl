# Exact first-stage bindings. No blanket clock groups or waived CDC findings.
proc ct_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc ct_pin {name} {
    set pins [get_pins -quiet $name]
    ct_one $pins $name
    if {[get_property NAME $pins] ne $name} {error "Non-exact pin lookup $name"}
    return $pins
}
proc ct_cells {pattern count} {
    set cells [get_cells -quiet -hier -regexp $pattern]
    if {[llength $cells]!=$count} {error "Expected $count cells matching $pattern, got [llength $cells]"}
    return $cells
}
proc ct_clock {cells expected} {
    set clocks [lsort -unique [get_clocks -of_objects [get_pins -of_objects $cells -filter {IS_CLOCK == 1}]]]
    if {$clocks ne $expected} {error "Clock mismatch $cells => $clocks, expected $expected"}
}
proc ct_groups {} {
    set groups {}
    for {set lane 0} {$lane<2} {incr lane} {
        foreach {fifo width wc rc} {commands 136 pcie_clk reader_clk returns 593 reader_clk pcie_clk} {
            foreach {a b sc dc} [list wg ws1 $wc $rc rg rs1 $rc $wc] {
                set from [ct_cells [format {^transport/lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $a] 3]
                set dest [ct_cells [format {^transport/lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $b] 3]
                ct_clock $from $sc;ct_clock $dest $dc
                lappend groups [list gray l${lane}_${fifo}_${a} $from [get_pins -of_objects $dest -filter {REF_PIN_NAME == D}] 4.0 2.0 3]
            }
            set dest [ct_cells [format {^transport/lanes\[%d\]\.%s/output_data_reg\[[0-9]+\]$} $lane $fifo] $width]
            set to [get_pins -of_objects $dest -filter {REF_PIN_NAME == D}]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}]
            set from [lsort -unique [get_cells -of_objects $drivers]]
            if {[llength $from]!=$width} {error "Capture RAM count changed"}
            foreach ram $from {if {[get_property REF_NAME $ram] ni {RAMD32 RAMS32}} {error "Unexpected capture driver $ram"}}
            ct_clock $from $wc;ct_clock $dest $rc
            lappend groups [list capture l${lane}_${fifo}_data $from $to 4.0 0 $width]
        }
    }
    foreach {src dst sc dc budget} {
        transport/request_reg {transport/echo_r_reg[0]/D} mgmt_clk reader_clk 4.0
        {transport/echo_r_reg[2]} {transport/echo_m_reg[0]/D} reader_clk mgmt_clk 4.0
        transport/reader_failed_reg {transport/failed_m_reg[0]/D} reader_clk mgmt_clk 4.0
        transport/armed_reg {transport/armed_p_reg[0]/D} mgmt_clk pcie_clk 4.0
        transport/management_fault_reg {transport/failed_p_reg[0]/D} mgmt_clk pcie_clk 4.0
        transport/management_fault_reg {transport/failed_r_reg[0]/D} mgmt_clk reader_clk 4.0
        transport/stop_held_reg {transport/stop_r_reg[0]/D} pcie_clk reader_clk 4.0
        reader/startup/started_reg {transport/started_m_reg[0]/D} reader_clk mgmt_clk 4.0
        endpoint/guard/request_reg {endpoint/guard/echo_a_reg[0]/D} mgmt_clk pcie_clk 4.0
        {endpoint/guard/echo_a_reg[2]} {endpoint/guard/echo_m_reg[0]/D} pcie_clk mgmt_clk 4.0
        endpoint/guard/armed_reg {endpoint/guard/armed_a_reg[0]/D} mgmt_clk pcie_clk 4.0
        memory_watch/request_reg {memory_watch/echo_mem_reg[0]/D} mgmt_clk mem_clk 2.222
        {memory_watch/echo_mem_reg[2]} {memory_watch/echo_m_reg[0]/D} mem_clk mgmt_clk 4.0
        memory_watch/armed_reg {memory_watch/armed_p_reg[0]/D} mgmt_clk pcie_clk 4.0
        memory_watch/fault_reg {memory_watch/failed_p_reg[0]/D} mgmt_clk pcie_clk 4.0
        memory_watch/fault_reg {memory_watch/failed_r_reg[0]/D} mgmt_clk reader_clk 4.0
        {reader/channels[0].adapter/tracker/fault_reg} {reader/startup/fault_meta_reg[0]/D} mem_clk reader_clk 4.0
        {reader/channels[1].adapter/tracker/fault_reg} {reader/startup/fault_meta_reg[1]/D} mem_clk reader_clk 4.0
        {reader/startup/memory_release_reg[2]} {reader/startup/ready_meta_reg[2]/D} mem_clk reader_clk 4.0
    } {
        set from [ct_one [get_cells -hier -quiet -filter [format {NAME == "%s"} $src]] $src]
        set to [ct_pin $dst];ct_clock $from $sc;ct_clock [get_cells -of_objects $to] $dc
        lappend groups [list control [string map {/ _ [ _ ] _ . _} $dst] $from $to $budget 0 1]
    }
    if {[llength $groups]!=31} {error "Expected 31 exact groups"}
    return $groups
}
proc ct_reset_arcs {} {
    set arcs {}
    for {set i 0} {$i<3} {incr i} {
        for {set bit 0} {$bit<3} {incr bit} {
            lappend arcs [list [format {endpoint/guard/availability[%d].release_m_reg[2]/C} $i] [format {endpoint/guard/availability[%d].release_a_reg[%d]/CLR} $i $bit]]
        }
        lappend arcs [list endpoint/guard/fault_reg/C [format {endpoint/guard/stop_a_reg[%d]/PRE} $i]]
    }
    return $arcs
}
proc ct_constrain {out} {
    set f [open $out/bindings.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tname\tsources\tendpoints\tmax_ns\tskew_ns"
    foreach group [ct_groups] {
        lassign $group kind name from to bound skew count
        puts $f "$kind\t$name\t[llength $from]\t[llength $to]\t$bound\t$skew"
        set_max_delay -datapath_only $bound -from $from -to $to
        if {$skew>0} {set_bus_skew $skew -from $from -to $to}
    };close $f
    set f [open $out/reset_arcs.tsv {WRONLY CREAT EXCL}];puts $f "from\tto\tdirect_driver"
    foreach pair [ct_reset_arcs] {
        lassign $pair a b;set from [ct_pin $a];set to [ct_pin $b]
        set driver [ct_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}] "$b driver"]
        set expected "[file dirname $a]/Q"
        if {$driver ne $expected} {
            set inv [get_cells -of_objects $driver]
            if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1"} {error "Unexpected async control driver $b: $driver"}
            set upstream [ct_one [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $inv/I0]] -filter {DIRECTION == OUT}] "$b inverter input"]
            if {$upstream ne $expected} {error "Async reset inverter has wrong source $b: $upstream"}
        }
        set cell [get_cells -of_objects $to]
        if {[get_property ASYNC_REG $cell] ni {1 TRUE true}} {error "Missing async reset synchronization $b"}
        set_false_path -from $from -to $to
        puts $f "$a\t$b\t$driver"
    };close $f
}
