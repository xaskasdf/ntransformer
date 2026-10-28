# Native composition bindings, adapted from the sealed clock-tree profile.
# Exact data bounds and async-control arcs; no clock groups or CDC waivers.
proc nc_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc nc_pin {name} {
    if {![regexp {^(core/|availability/|ntrpu_bridge_i/)} $name]} {set name core/$name}
    set pins [get_pins -quiet $name]
    nc_one $pins $name
    if {[get_property NAME $pins] ne $name} {error "Non-exact pin lookup $name"}
    return $pins
}
proc nc_cells {pattern count} {
    set pattern [string replace $pattern 0 0 {^core/}]
    set cells [get_cells -quiet -hier -regexp $pattern]
    if {[llength $cells]!=$count} {error "Expected $count cells matching $pattern, got [llength $cells]"}
    return $cells
}
proc nc_clock {cells expected} {
    set expected [dict get $::nc_clocks $expected]
    set clocks [lsort -unique [get_clocks -of_objects [get_pins -of_objects $cells -filter {IS_CLOCK == 1}]]]
    if {$clocks ne $expected} {error "Clock mismatch $cells => $clocks, expected $expected"}
}
proc nc_groups {} {
    set groups {}
    for {set lane 0} {$lane<2} {incr lane} {
        foreach {fifo width wc rc} {commands 136 pcie_clk reader_clk returns 593 reader_clk pcie_clk} {
            foreach {a b sc dc} [list wg ws1 $wc $rc rg rs1 $rc $wc] {
                set from [nc_cells [format {^transport/lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $a] 3]
                set dest [nc_cells [format {^transport/lanes\[%d\]\.%s/%s_reg\[[0-2]\]$} $lane $fifo $b] 3]
                nc_clock $from $sc;nc_clock $dest $dc
                lappend groups [list gray l${lane}_${fifo}_${a} $from [get_pins -of_objects $dest -filter {REF_PIN_NAME == D}] 4.0 2.0 3]
            }
            set dest [nc_cells [format {^transport/lanes\[%d\]\.%s/output_data_reg\[[0-9]+\]$} $lane $fifo] $width]
            set to [get_pins -of_objects $dest -filter {REF_PIN_NAME == D}]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}]
            set from [lsort -unique [get_cells -of_objects $drivers]]
            if {[llength $from]!=$width} {error "Capture RAM count changed"}
            foreach ram $from {if {[get_property REF_NAME $ram] ni {RAMD32 RAMS32 RAMD64E RAMS64E}} {error "Unexpected capture driver $ram"}}
            nc_clock $from $wc;nc_clock $dest $rc
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
        set src core/$src
        set from [nc_one [get_cells -hier -quiet -filter [format {NAME == "%s"} $src]] $src]
        set to [nc_pin $dst];nc_clock $from $sc;nc_clock [get_cells -of_objects $to] $dc
        lappend groups [list control [string map {/ _ [ _ ] _ . _} $dst] $from $to $budget 0 1]
    }
    for {set i 0} {$i<2} {incr i} {
        set from [nc_one [get_cells -hier -quiet -filter [format {NAME == "availability/ready_mgmt_reg[%d]"} $i]] ready]
        set to [nc_pin [format {reader/startup/ready_meta_reg[%d]/D} $i]]
        nc_clock $from mgmt_clk;nc_clock [get_cells -of_objects $to] reader_clk
        lappend groups [list control availability_$i $from $to 4.0 0 1]
    }
    if {[llength $groups]!=33} {error "Expected 33 exact groups"}
    return $groups
}
proc nc_reset_arcs {} {
    set arcs {}
    for {set i 0} {$i<3} {incr i} {
        for {set bit 0} {$bit<3} {incr bit} {
            lappend arcs [list [format {endpoint/guard/availability[%d].release_m_reg[2]/C} $i] [format {endpoint/guard/availability[%d].release_a_reg[%d]/CLR} $i $bit]]
        }
        lappend arcs [list endpoint/guard/fault_reg/C [format {endpoint/guard/stop_a_reg[%d]/PRE} $i]]
    }
    foreach {n reg} {0 user_lnk_up_reg 1 user_reset_reg} {
        for {set i 0} {$i<3} {incr i} {
            lappend arcs [list ntrpu_bridge_i/inst/pcie4c_ip_i/inst/$reg/C [format {endpoint/guard/availability[%d].release_m_reg[%d]/CLR} $n $i]]
        }
    }
    return $arcs
}
proc nc_constrain {out {apply 1}} {
    set f [open $out/bindings.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tname\tsources\tendpoints\tmax_ns\tskew_ns"
    foreach group [nc_groups] {
        lassign $group kind name from to bound skew count
        puts $f "$kind\t$name\t[llength $from]\t[llength $to]\t$bound\t$skew"
        if {$apply} {
            set_max_delay -datapath_only $bound -from $from -to $to
            if {$skew>0} {set_bus_skew $skew -from $from -to $to}
        }
    };close $f
    set f [open $out/reset_arcs.tsv {WRONLY CREAT EXCL}];puts $f "from\tto\tdirect_driver\tsource_q\tlut_cells\tlut_inits\tinversions"
    foreach pair [nc_reset_arcs] {
        lassign $pair a b;set from [nc_pin $a];set to [nc_pin $b]
        set driver [nc_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $to] -filter {DIRECTION == OUT}] "$b driver"]
        set expected "[file dirname [get_property NAME $from]]/Q"
        # Exact, bounded chain from destination to source. Native AXI reset has
        # two inverters around the IP's explicit LUT1 buffer (INIT=2'h2).
        set chain {};set inits {};set inversions 0
        if {[regexp {^core/endpoint/guard/availability\[([0-2])\]\.release_([am])_reg\[[0-2]\]/CLR$} $to -> n domain]} {
            lappend chain [format {core/endpoint/guard/availability[%d].release_%s[2]_i_1} $n $domain]
            lappend inits "2'h1"
            if {$domain eq "m" && $n==1} {
                lappend chain ntrpu_bridge_i/inst/axi_aresetn_inst ntrpu_bridge_i/inst/axi_aresetn_inferred_i_1
                lappend inits "2'h2" "2'h1"
            }
        }
        set upstream $driver
        foreach lut $chain init $inits {
            if {$upstream ne "$lut/O"} {error "Unexpected async control chain $b: $upstream, expected $lut/O"}
            set cell [get_cells -of_objects $upstream]
            if {[get_property REF_NAME $cell] ne "LUT1" || [get_property INIT $cell] ne $init} {error "Async control function changed $lut"}
            if {$init eq "2'h1"} {incr inversions}
            set upstream [nc_one [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $lut/I0]] -filter {DIRECTION == OUT}] "$b LUT input"]
        }
        if {$upstream ne $expected} {error "Async reset chain has wrong source $b: $upstream"}
        set cell [get_cells -of_objects $to]
        if {[get_property ASYNC_REG $cell] ni {1 TRUE true}} {error "Missing async reset synchronization $b"}
        if {$apply} {set_false_path -from $from -to $to}
        puts $f "[get_property NAME $from]\t[get_property NAME $to]\t$driver\t$upstream\t[join $chain |]\t[join $inits |]\t$inversions"
    };close $f
}
