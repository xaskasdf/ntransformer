# Exact constraints and connectivity for the new native control composition.
# Existing native groups are re-derived and checked; no global clock cuts.
source [file join [file dirname [info script]] audit_hbm_native_boundary.tcl]
proc bc_driver {pin} {
    set nets [get_nets -quiet -segments -of_objects $pin]
    if {![llength $nets]} {return {}}
    return [lsort -unique [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]]
}
proc bc_connect {f a b} {
    set pa [nb_pin $a];set pb [nb_pin $b]
    set da [bc_driver $pa];set db [bc_driver $pb]
    if {$da ne $db || [llength $da]!=1} {error "Connection mismatch $a => $b: $da / $db"}
    puts $f "$a\t$b\t$da"
}
proc bc_structure {out} {
    set f [open $out/csr_connections.tsv {WRONLY CREAT EXCL}];puts $f "adapter\tcontrol\tdriver"
    foreach {name width} {awaddr 12 araddr 12 awprot 3 arprot 3 awvalid 1 wvalid 1 bready 1 arvalid 1 rready 1 wdata 32 wstrb 4 awready 1 wready 1 arready 1 bvalid 1 rvalid 1 bresp 2 rresp 2 rdata 32} {
        for {set bit 0} {$bit<$width} {incr bit} {
            set suffix [expr {$width==1 ? "" : [format {[%d]} $bit]}]
            bc_connect $f bar_control/m_${name}${suffix} control/s_${name}${suffix}
        }
    };close $f
    set f [open $out/bridge_connections.tsv {WRONLY CREAT EXCL}];puts $f "bridge\tadapter\tdriver"
    foreach {name width} {awaddr 64 araddr 64 awid 4 arid 4 awlen 8 arlen 8 awsize 3 arsize 3 awprot 3 arprot 3 awburst 2 arburst 2 awvalid 1 wvalid 1 wlast 1 bready 1 arvalid 1 rready 1 wdata 512 wstrb 64 awready 1 wready 1 arready 1 bid 4 rid 4 bresp 2 rresp 2 bvalid 1 rvalid 1 rlast 1 rdata 512} {
        for {set bit 0} {$bit<$width} {incr bit} {
            set suffix [expr {$width==1 ? "" : [format {[%d]} $bit]}]
            bc_connect $f ntrpu_bridge_i/m_axib_${name}${suffix} bar_control/s_${name}${suffix}
        }
    };close $f
    # This frozen synthesis folds the two LOCK outputs across the IP boundary.
    # They are not live wires: require disconnected IP pins AND literal GND at
    # the adapter. No other missing bridge connection receives this exception.
    set f [open $out/bridge_constants.tsv {WRONLY CREAT EXCL}]
    puts $f "bridge\tadapter\tbridge_leaf_drivers\tadapter_driver\tadapter_ref"
    foreach name {awlock arlock} {
        set a ntrpu_bridge_i/m_axib_$name;set b bar_control/s_$name
        set da [bc_driver [nb_pin $a]];set db [nc_one [bc_driver [nb_pin $b]] lock_constant]
        set ref [get_property REF_NAME [get_cells -of_objects [get_pins $db]]]
        if {[llength $da] || $ref ne "GND"} {error "Unexpected native LOCK binding $a => $b: $da / $db ($ref)"}
        puts $f "$a\t$b\t[llength $da]\t$db\t$ref"
    };close $f
    set f [open $out/cold_reset_bindings.tsv {WRONLY CREAT EXCL}];puts $f "source\tconsumer\tdriver"
    bc_connect $f core/client_cold_rst_n control/cold_rst_n
    bc_connect $f ntrpu_bridge_i/axi_aresetn control/bus_rst_n
    foreach domain {0 1 2 3} source {client_cold_rst_n management_cold_rst_n reader_cold_rst_n mem_cold_rst_n} {
        set tap [format {control/snapshot/taps[%d].tap} $domain]
        bc_connect $f core/$source $tap/src_cold_rst_n
        bc_connect $f core/client_cold_rst_n $tap/dst_cold_rst_n
    };close $f
    set f [open $out/physical_ports.tsv {WRONLY CREAT EXCL}];puts $f "port\tdirection"
    foreach p [lsort [get_ports]] {puts $f "$p\t[get_property DIRECTION [get_ports $p]]"}
    close $f
    if {[llength [get_ports]]!=74} {error "Expected only 74 physical/sequencer port bits"}
    set f [open $out/telemetry_capture_sources.tsv {WRONLY CREAT EXCL}];puts $f "domain\tcapture\tsource_clock\tsource_pin"
    foreach domain {0 1 2 3} logical {pcie_clk mgmt_clk reader_clk mem_clk} {
        set expected [dict get $::nc_clocks $logical]
        set held [get_cells -quiet -hier -regexp [format {^control/snapshot/taps\[%d\]\.tap/held_data_reg(\[[0-9]+\])?$} $domain]]
        foreach cell $held {
            set dp [nb_pin $cell/D]
            foreach {other clock} $::nc_clocks {
                set path [get_timing_paths -quiet -from [get_clocks $clock] -to $dp -max_paths 1]
                if {[llength $path]} {
                    if {$clock ne $expected} {error "Telemetry sampled from wrong domain: $cell from $clock"}
                    puts $f "$domain\t$cell\t$clock\t[get_property STARTPOINT_PIN $path]"
                }
            }
        }
    };close $f
}
proc bc_snapshot {out apply} {
    set groups {};set s [open $out/snapshot_interstage.tsv {WRONLY CREAT EXCL}]
    puts $s "from\tto\ttype\trequirement_ns\tslack_ns"
    foreach domain {0 1 2 3} logical {pcie_clk mgmt_clk reader_clk mem_clk} {
        set tap [format {control/snapshot/taps[%d].tap} $domain]
        set clock [dict get $::nc_clocks $logical]
        foreach chain {request_sync ack_sync} expected [list $clock [dict get $::nc_clocks pcie_clk]] {
            for {set bit 0} {$bit<3} {incr bit} {
                set cell [nc_one [get_cells -quiet [format {%s/%s_reg[%d]} $tap $chain $bit]] snapshot_sync]
                if {[get_property ASYNC_REG $cell] ni {1 TRUE true} || [get_clocks -of_objects [get_pins $cell/C]] ne $expected} {error "Snapshot synchronizer changed: $cell"}
                if {$bit<2} {
                    set to [nb_pin [format {%s/%s_reg[%d]/D} $tap $chain [expr {$bit+1}]]]
                    foreach type {max min} {
                        set path [nc_one [get_timing_paths -quiet -from [get_pins $cell/C] -to $to -delay_type $type -max_paths 1] interstage]
                        if {[get_property STARTPOINT_CLOCK $path] ne $expected || [get_property ENDPOINT_CLOCK $path] ne $expected} {error "Interstage clock mismatch"}
                        puts $s "$cell\t$to\t$type\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
                    }
                }
            }
        }
        # Domain zero shares the actual PCIe clock, so retain synchronous timing.
        if {$domain==0} {continue}
        set req [nb_pin [format {%s/request_sync_reg[0]/D} $tap]]
        set ack [nb_pin [format {%s/ack_sync_reg[0]/D} $tap]]
        set request_bound [expr {$domain==3?2.222:4.0}]
        lappend groups [list token request_$domain [get_cells control/snapshot/token_reg] [list $req] $request_bound]
        lappend groups [list token ack_$domain [get_cells $tap/ack_token_reg] [list $ack] 4.0]
        set samples [get_cells -quiet -hier -regexp [format {^control/snapshot/taps\[%d\]\.tap/sample_reg(\[[0-9]+\])?$} $domain]]
        set from {};set to {}
        foreach cell $samples {
            set dp [nb_pin $cell/D]
            set driver [nc_one [bc_driver $dp] snapshot_data_driver]
            set source [get_cells -of_objects $driver]
            if {[get_property REF_NAME $source] in {VCC GND}} {continue}
            if {[string first "$tap/held_data_reg" $source]!=0} {error "Unexpected snapshot data driver: $driver => $dp"}
            if {[get_clocks -of_objects [get_pins $source/C]] ne $clock} {error "Wrong held-data source clock"}
            if {[get_clocks -of_objects [get_pins $cell/C]] ne [dict get $::nc_clocks pcie_clk]} {error "Wrong sample clock"}
            lappend from $source;lappend to $dp
        }
        if {![llength $to]} {error "Snapshot bus optimized away for domain $domain"}
        lappend groups [list held_data data_$domain [lsort -unique $from] $to 4.0]
    };close $s
    set f [open $out/snapshot_bounds.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tgroup\tfrom\tto\trequirement_ns\tslack_ns"
    foreach group $groups {
        lassign $group kind name from to bound
        if {$apply} {set_max_delay -datapath_only $bound -from $from -to $to}
        set paths [get_timing_paths -quiet -from $from -to $to -delay_type max -max_paths [llength $to] -nworst 1]
        if {[llength $paths]!=[llength $to]} {error "Missing snapshot paths $name"}
        set actual {}
        foreach path $paths {
            set endpoint [get_property ENDPOINT_PIN $path];lappend actual $endpoint
            if {abs([get_property REQUIREMENT $path]-$bound)>0.001} {error "Snapshot limit overridden $name"}
            puts $f "$kind\t$name\t[get_property STARTPOINT_PIN $path]\t$endpoint\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
        }
        if {[lsort $actual] ne [lsort $to]} {error "Snapshot endpoint set changed $name"}
    };close $f
}
proc bc_run {checkpoint out apply} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled"} {error "Wrong native top"}
    nc_clock_map $out
    nc_constrain $out $apply
    nc_topology $out
    nc_paths $out
    nb_lock $out $apply
    bc_structure $out
    bc_snapshot $out $apply
    report_cdc -details -file $out/cdc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_exceptions -ignored -file $out/ignored.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    write_xdc -constraints all $out/effective.xdc
    if {$apply} {write_checkpoint $out/bar_bounded.dcp}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f BAR_NATIVE_CONTROL_REVIEW_REQUIRED;close $f
    close_design
}
