# Scoped loader bounds in the complete native serial composition.
# Reuse all previous audits; adapt only the two CSR segments through the loader.
source [file join [file dirname [info script]] audit_hbm_bar_control.tcl]
rename bc_connect wp_original_connect
proc bc_connect {f a b} {
    if {[regexp {^bar_control/m_(.+)$} $a -> suffix] && $b eq "control/s_$suffix"} {
        wp_original_connect $f $a weight_loader/s_$suffix
        wp_original_connect $f weight_loader/m_$suffix $b
    } else {wp_original_connect $f $a $b}
}
proc wp_cell {name} {
    set c [nc_one [get_cells -quiet $name] $name]
    if {[get_property NAME $c] ne $name} {error "Non-exact loader cell $name"}
    return $c
}
proc wp_sync {out} {
    set f [open $out/loader_synchronizers.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tclock\tasync_reg"
    set t [open $out/loader_interstage.tsv {WRONLY CREAT EXCL}]
    puts $t "from\tto\ttype\tstart_clock\tend_clock\trequirement_ns\tslack_ns"
    set actual [get_cells -quiet -hier -filter {NAME =~ weight_loader/* && ASYNC_REG == TRUE}]
    if {[llength $actual]!=12} {error "Expected twelve loader synchronizer registers"}
    foreach chain {req_sync mem_ready_sync ack_sync ready_sync} domain {mem_clk mem_clk pcie_clk pcie_clk} {
        set clock [dict get $::nc_clocks $domain]
        for {set bit 0} {$bit<3} {incr bit} {
            set cell [wp_cell [format {weight_loader/%s_reg[%d]} $chain $bit]]
            if {[lsearch -exact $actual $cell]<0} {error "Unmarked loader synchronizer $cell"}
            nc_clock $cell $domain
            puts $f "$cell\t$clock\t[get_property ASYNC_REG $cell]"
            if {$bit==2} {continue}
            set to [nb_pin [format {weight_loader/%s_reg[%d]/D} $chain [expr {$bit+1}]]]
            set loads [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects [get_pins $cell/Q]] -filter {DIRECTION == IN}]
            if {$loads ne $to} {error "Loader intermediate-stage fanout $cell: $loads"}
            foreach type {max min} {
                set path [nc_one [get_timing_paths -quiet -from [get_pins $cell/C] -to $to -delay_type $type -max_paths 1] interstage]
                if {[get_property STARTPOINT_CLOCK $path] ne $clock || [get_property ENDPOINT_CLOCK $path] ne $clock} {error "Loader interstage clock mismatch"}
                puts $t "$cell\t$to\t$type\t$clock\t$clock\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
            }
        }
    };close $f;close $t
}
proc wp_connections {out} {
    set f [open $out/loader_clock_reset.tsv {WRONLY CREAT EXCL}]
    puts $f "source\tconsumer\tdriver"
    foreach {a b} {
        core/clk weight_loader/clk
        core/mem_clk weight_loader/mem_clk
        core/client_cold_rst_n weight_loader/cold_rst_n
        core/mem_cold_rst_n weight_loader/mem_cold_rst_n
        ntrpu_bridge_i/axi_aresetn weight_loader/bus_rst_n
        core/mem_clk hbm_binding/axi_clk
        core/mem_cold_rst_n hbm_binding/axi_cold_reset_n
    } {wp_original_connect $f $a $b};close $f
    set f [open $out/loader_hbm_connections.tsv {WRONLY CREAT EXCL}]
    puts $f "loader\tnative\tloader_driver\tnative_driver\tconstant"
    foreach lane {0 1} pc {00 16} {
        foreach {name width value} {awaddr 34 -1 awburst 2 1 awid 6 0 awlen 4 0 awsize 3 5 awvalid 1 -1 awready 1 -1 wdata 256 -1 wstrb 32 4294967295 wlast 1 1 wvalid 1 -1 wready 1 -1 bid 6 -1 bresp 2 -1 bvalid 1 -1 bready 1 -1} {
            for {set bit 0} {$bit<$width} {incr bit} {
                set a [format {weight_loader/h_%s[%d]} $name [expr {$lane*$width+$bit}]]
                set suffix [expr {$width==1 ? "" : [format {[%d]} $bit]}]
                set b hbm_binding/memory/AXI_${pc}_[string toupper $name]$suffix
                set da [bc_driver [nb_pin $a]];set db [nc_one [bc_driver [nb_pin $b]] native_hbm_driver]
                if {[llength $da]==1} {set da [lindex $da 0]}
                set constant -1
                if {$value>=0} {
                    set constant [expr {($value>>$bit)&1}]
                    set expected [expr {$constant ? "VCC" : "GND"}]
                    if {[get_property REF_NAME [get_cells -of_objects [get_pins $db]]] ne $expected} {error "Wrong native HBM constant $b"}
                    if {[llength $da] && $da ne $db} {error "Wrong loader constant $a"}
                } elseif {$da ne $db || [llength $da]!=1} {error "Native write binding differs: $a => $b: $da / $db"}
                puts $f "$a\t$b\t$da\t$db\t$constant"
            }
        }
    };close $f
}
proc wp_bounds {out apply} {
    set groups {}
    set from {};set to {}
    # Synthesis merges address[33] and lane; all other data bits stay individual.
    for {set bit 0} {$bit<34} {incr bit} {
        lappend from [wp_cell [format {weight_loader/active_reg[%d]} $bit]]
        if {$bit==33} {set dest weight_loader/lane_reg/D} else {set dest [format {weight_loader/address_reg[%d]/D} $bit]}
        lappend to [nb_pin $dest]
    }
    for {set bit 0} {$bit<256} {incr bit} {
        lappend from [wp_cell [format {weight_loader/active_reg[%d]} [expr {$bit+128}]]]
        lappend to [nb_pin [format {weight_loader/payload_reg[%d]/D} $bit]]
    }
    foreach src $from dst $to {
        if {[nc_one [bc_driver $dst] loader_data_driver] ne "$src/Q"} {error "Loader held-bus capture changed $src => $dst"}
    }
    lappend groups [list payload $from $to pcie_clk mem_clk 2.222 1.111]
    set from {};set to {}
    for {set bit 0} {$bit<4} {incr bit} {
        set src [wp_cell [format {weight_loader/memory_result_reg[%d]} $bit]]
        set dst [nb_pin [format {weight_loader/completion_reg[%d]/D} $bit]]
        if {[nc_one [bc_driver $dst] loader_completion_driver] ne "$src/Q"} {error "Loader completion capture changed"}
        lappend from $src;lappend to $dst
    }
    lappend groups [list completion $from $to mem_clk pcie_clk 4.0 2.0]
    # The sticky failure reduction consumes all four bits too. Bound each
    # source-to-failed path, not just the four visible completion registers.
    foreach src $from {
        lappend groups [list failure_[string map {[ _ ] _} [file tail $src]] [list $src] [list [nb_pin weight_loader/failed_reg/D]] mem_clk pcie_clk 4.0 0]
    }
    foreach {src dest sc dc bound} {
        request_reg {req_sync_reg[0]/D} pcie_clk mem_clk 2.222
        acknowledge_reg {ack_sync_reg[0]/D} mem_clk pcie_clk 4.0
        {mem_ready_sync_reg[2]} {ready_sync_reg[0]/D} mem_clk pcie_clk 4.0
    } {
        set src [wp_cell weight_loader/$src];set dst [nb_pin weight_loader/$dest]
        if {[nc_one [bc_driver $dst] loader_token_driver] ne "$src/Q"} {error "Loader token source changed"}
        lappend groups [list token_[string map {[ _ ] _ / _} $dest] [list $src] [list $dst] $sc $dc $bound 0]
    }
    set ready [nb_pin {weight_loader/mem_ready_sync_reg[0]/D}]
    set starts [lsort [all_fanin -flat -startpoints_only -to $ready]]
    # all_fanin reports timing startpoints (C), rather than net drivers (Q).
    set expected [lsort [list {availability/ready_mgmt_reg[0]/C} {availability/ready_mgmt_reg[1]/C} availability/fault_reg/C]]
    if {$starts ne $expected} {error "Unexpected readiness fan-in: $starts"}
    set f [open $out/loader_readiness_fanin.tsv {WRONLY CREAT EXCL}]
    puts $f "source\tto\tdirect_driver"
    foreach pin $starts {
        puts $f "$pin\t$ready\t[bc_driver $ready]"
        set cell [wp_cell [file dirname $pin]]
        lappend groups [list readiness_[string map {[ _ ] _} [file tail $cell]] [list $cell] [list $ready] mgmt_clk mem_clk 2.222 0]
    };close $f
    set f [open $out/loader_bounds.tsv {WRONLY CREAT EXCL}]
    puts $f "group\tfrom\tto\tstart_clock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    set g [open $out/loader_groups.tsv {WRONLY CREAT EXCL}]
    puts $g "group\tsources\tendpoints\tmax_ns\tskew_ns"
    foreach group $groups {
        lassign $group name from to sc dc bound skew
        nc_clock $from $sc;nc_clock [get_cells -of_objects $to] $dc
        puts $g "$name\t[llength $from]\t[llength $to]\t$bound\t$skew"
        if {$apply} {
            set_max_delay -datapath_only $bound -from $from -to $to
            if {$skew>0} {set_bus_skew $skew -from $from -to $to}
        }
        set paths [get_timing_paths -quiet -from $from -to $to -delay_type max -max_paths [llength $to] -nworst 1]
        if {[llength $paths]!=[llength $to]} {error "Missing loader paths $name"}
        set seen {}
        foreach path $paths {
            set dst [get_property ENDPOINT_PIN $path];lappend seen $dst
            if {abs([get_property REQUIREMENT $path]-$bound)>0.001} {error "Overridden loader bound $name"}
            puts $f [join [list $name [get_property STARTPOINT_PIN $path] $dst [get_property STARTPOINT_CLOCK $path] [get_property ENDPOINT_CLOCK $path] [get_property REQUIREMENT $path] [get_property SLACK $path] [get_property DATAPATH_DELAY $path]] "\t"]
        }
        if {[lsort $seen] ne [lsort $to]} {error "Loader path coverage changed $name"}
    };close $f;close $g
}
proc wp_run {checkpoint out apply} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" || [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong physical loader donor"}
    write_xdc -constraints all $out/before.xdc
    nc_clock_map $out;nc_constrain $out $apply;nc_topology $out;nc_paths $out
    nb_lock $out $apply;bc_structure $out;bc_snapshot $out $apply
    wp_connections $out;wp_bounds $out $apply;wp_sync $out
    report_cdc -details -file $out/cdc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_exceptions -ignored -file $out/ignored.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_drc -file $out/drc.rpt
    write_xdc -constraints all $out/effective.xdc
    if {$apply} {write_checkpoint $out/weight_bounded.dcp}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f WEIGHT_NATIVE_PHYSICAL_REVIEW_REQUIRED;close $f
    close_design
}
