# Timing experiment only: neighboring launch/capture FFs, not a protocol adapter.
set root [file normalize [file join [file dirname [info script]] ..]]
proc boundary_clock {name direction} {
    set base [lindex [split $name {[}] 0]
    if {$direction eq "IN"} {
        set dma [expr {$base in {dma_link_up dma_run dma_error dma_done dsc_ready axis_ready}}]
    } elseif {$direction eq "OUT"} {
        set dma [expr {$base in {dma_rst_n dma_fault dsc_load axis_valid axis_last dsc_src dsc_dst axis_keep dsc_len dsc_ctl axis_data}}]
    } else {error "Unsupported port direction"}
    return [expr {$dma ? "dma_clk" : "reader_clk"}]
}
proc leaf_pins {net direction} {
    return [lsort [get_pins -quiet -leaf -of_objects [get_nets -segments $net] -filter [list DIRECTION == $direction]]]
}
proc original_pins {net direction} {
    set result {}
    foreach p [leaf_pins $net $direction] {
        if {![string match ntrpu_boundary_* $p]} {lappend result $p}
    }
    return $result
}
proc reset_scope {out constrain} {
    set raw [leaf_pins [get_nets -of_objects [get_ports cold_reset_n]] IN]
    set cold [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]
    if {[llength $cold]!=10} {error "Unexpected cold-reset scope"}
    if {[llength $raw]==1} {
        set inv [get_cells -of_objects $raw]
        if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1"} {error "Unexpected reset inversion"}
        set op [get_pins -of_objects $inv -filter {DIRECTION == OUT}]
        if {[leaf_pins [get_nets -of_objects $op] IN] ne [lsort $cold]} {error "Reset inversion is not directly connected to startup CLR pins"}
        set inverted 0;set representation LUT1
    } elseif {[lsort $raw] eq [lsort $cold]} {
        set inverted 1;set representation CLR_ATTRIBUTE
    } else {error "Unexpected raw reset loads"}
    set f [open $out/reset_polarity.tsv w]
    puts $f "endpoint\trepresentation\tclr_inverted\tclock\tref\tasync_reg\tinit"
    foreach p $cold {
        set c [get_cells -of_objects $p]
        if {[get_property REF_PIN_NAME $p] ne "CLR" || [get_property REF_NAME $c] ne "FDCE" || ![get_property ASYNC_REG $c]} {error "Reset reaches unexpected endpoint"}
        set v [get_property IS_CLR_INVERTED $c]
        if {$v in {1 1'b1 true TRUE}} {set bit 1} elseif {$v in {0 1'b0 false FALSE}} {set bit 0} else {error "Unknown CLR polarity: $c = $v"}
        if {$bit!=$inverted || [get_property INIT $c] ne "1'b0"} {error "Cold reset is not active-low/initially clear"}
        set clock [get_clocks -of_objects [get_pins -of_objects $c -filter {REF_PIN_NAME == C}]]
        set expected [expr {[string match dr_reg* $c] ? "dma_clk" : "reader_clk"}]
        if {$clock ne $expected} {error "Reset synchronizer clock changed"}
        puts $f "$p\t$representation\t$bit\t$clock\t[get_property REF_NAME $c]\t[get_property ASYNC_REG $c]\t[get_property INIT $c]"
    }
    close $f
    if {$constrain} {set_false_path -from [get_ports cold_reset_n] -to $cold}
    set f [open $out/cold_reset_endpoints.txt w];puts $f [join [lsort $cold] \n];close $f
}
proc insert_boundary {out} {
    create_cell -reference GND ntrpu_boundary_gnd_source
    create_net ntrpu_boundary_gnd
    connect_net -net ntrpu_boundary_gnd -objects [get_pins ntrpu_boundary_gnd_source/G]
    set clknet(reader_clk) [get_nets -of_objects [get_pins ntrpu_phy_reader_bufg/O]]
    set clknet(dma_clk) [get_nets -of_objects [get_pins ntrpu_phy_dma_bufg/O]]
    set inputs {};set outputs {};set launch_d {};set capture_cells {};set rows {};set anchors {}
    set clock_pins(reader_clk) {};set clock_pins(dma_clk) {}
    foreach clock {reader_clk dma_clk} {set clock_inputs($clock) {};set clock_outputs($clock) {}}
    set f [open $out/boundary.tsv w]
    puts $f "port\tdirection\tclock\tanchor\tcore_net\toriginal_pins"
    set index 0
    foreach p [lsort [get_ports]] {
        set name [get_property NAME $p]
        if {$name in {clk dma_clk cold_reset_n}} {continue}
        set direction [get_property DIRECTION $p]
        set clock [boundary_clock $name $direction]
        set a [format ntrpu_boundary_ff_%05d $index];incr index
        set net [get_nets -quiet -of_objects $p]
        if {[llength $net]>1} {error "Ambiguous port net: $p"}
        if {![llength $net]} {
            if {$direction ne "IN"} {error "Unconnected output: $p"}
            create_net ${a}_core;set net [get_nets ${a}_core]
            connect_net -net $net -objects $p
        }
        set pd [expr {$direction eq "IN" ? "IN" : "OUT"}]
        set before [original_pins $net $pd]
        create_cell -reference FDRE $a
        lappend anchors $a
        lappend clock_pins($clock) $a/C
        create_net ${a}_outer
        disconnect_net -net $net -objects $p
        if {$direction eq "IN"} {
            connect_net -net ${a}_outer -objects [concat $p [get_pins $a/D]]
            connect_net -net $net -objects [get_pins $a/Q]
            lappend inputs $p;lappend launch_d [get_pins $a/D]
            lappend clock_inputs($clock) $p
        } else {
            connect_net -net $net -objects [get_pins $a/D]
            connect_net -net ${a}_outer -objects [concat $p [get_pins $a/Q]]
            lappend outputs $p;lappend capture_cells [get_cells $a]
            lappend clock_outputs($clock) $p
        }
        if {$before ne [original_pins $net $pd]} {error "Core boundary connectivity changed: $name"}
        lappend rows [list $name $direction $clock $a $net $before]
        puts $f "$name\t$direction\t$clock\t$a\t$net\t[join $before |]"
        if {$name eq "dma_link_up"} {set link_source [get_cells $a]}
    }
    close $f
    if {$index!=11375 || [llength $inputs]!=7662 || [llength $outputs]!=3713} {error "Wrong boundary size"}
    set all [get_cells $anchors]
    set_property DONT_TOUCH true $all
    set_property INIT 1'b0 $all
    connect_net -net ntrpu_phy_vcc -objects [get_pins -of_objects $all -filter {REF_PIN_NAME == CE}]
    connect_net -net ntrpu_boundary_gnd -objects [get_pins -of_objects $all -filter {REF_PIN_NAME == R}]
    foreach clock {reader_clk dma_clk} {
        connect_net -net $clknet($clock) -objects [get_pins $clock_pins($clock)]
        set_input_delay -clock [get_clocks $clock] -min 0.1 $clock_inputs($clock)
        set_input_delay -clock [get_clocks $clock] -max 1.0 $clock_inputs($clock)
        set_output_delay -clock [get_clocks $clock] -min -0.1 $clock_outputs($clock)
        set_output_delay -clock [get_clocks $clock] -max 1.0 $clock_outputs($clock)
    }
    write_checkpoint $out/anchors_unconstrained.dcp
    # Cut only the unmodelled measurement apparatus, outside the launch/capture
    # FFs. Every core-facing Q->core and core->D path keeps its timing checks.
    set_false_path -from $inputs -to $launch_d
    set_false_path -from $capture_cells -to $outputs
    set_max_delay -datapath_only 4.0 -from $link_source -to [get_pins link_s1_reg/D]
    # Recheck all original consumers/drivers after every insertion, including
    # aliases shared by multiple outputs. No original core pin may be lost.
    set f [open $out/connectivity.tsv w];puts $f "port\tunchanged_original_pins"
    foreach row $rows {
        lassign $row name direction clock a net before
        if {$before ne [original_pins $net $direction]} {error "Final ECO changed original pins: $name"}
        puts $f "$name\t[join $before |]"
    }
    close $f
}
proc audit_boundary {manifest out} {
    set f [open $manifest];set lines [split [read $f] \n];close $f
    set result [open $out/anchors.tsv w]
    puts $result "port\tdirection\tclock\tanchor\tloc\tcore_pins\tcore_route\touter_route\tcore_ports"
    set count 0
    foreach line [lrange $lines 1 end] {
        if {[string trim $line] eq ""} {continue}
        lassign [split $line \t] name direction clock a net original
        set c [get_cells $a]
        if {[llength $c]!=1 || [get_property REF_NAME $c] ne "FDRE" || ![get_property DONT_TOUCH $c]} {error "Lost boundary register: $a"}
        if {[get_clocks -of_objects [get_pins $a/C]] ne $clock} {error "Boundary clock changed: $a"}
        foreach pin {CE R} expected {VCC GND} {
            set drivers [get_cells -of_objects [leaf_pins [get_nets -of_objects [get_pins $a/$pin]] OUT]]
            if {[llength $drivers]!=1 || [get_property REF_NAME $drivers] ne $expected} {error "Boundary enable/reset changed: $a"}
        }
        set pn [get_nets -of_objects [get_ports [list $name]]]
        if {$direction eq "IN"} {
            if {[leaf_pins $pn IN] ne "$a/D"} {error "Input bypasses launch register: $name"}
            set cn [get_nets -of_objects [get_pins $a/Q]]
            set cp [leaf_pins $cn IN]
        } else {
            if {[leaf_pins $pn OUT] ne "$a/Q"} {error "Output bypasses capture register: $name"}
            set cn [get_nets -of_objects [get_pins $a/D]]
            set cp [leaf_pins $cn OUT]
        }
        set core_ports [get_ports -quiet -of_objects [get_nets -segments $cn]]
        if {[llength $core_ports]} {error "Core-facing boundary net still reaches an OOC port: $name"}
        if {$name eq "dma_link_up"} {set link $c}
        puts $result "$name\t$direction\t$clock\t$a\t[get_property LOC $c]\t[join $cp |]\t[get_property ROUTE_STATUS $cn]\t[get_property ROUTE_STATUS $pn]\t$core_ports"
        incr count
    }
    close $result
    if {$count!=11375} {error "Incomplete anchor audit"}
    set d [get_pins link_s1_reg/D]
    if {[leaf_pins [get_nets -of_objects $d] OUT] ne "$link/Q"} {error "Link no longer directly synchronized"}
    report_timing -from $link -to $d -max_paths 1 -input_pins -file $out/link_timing.rpt
}
proc export_boundary {root out manifest} {
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" ||
        [llength [get_clocks]]!=2 || [llength [get_cells -hier -filter {REF_NAME =~ BUFG*}]]!=2} {
        error "Changed device or clock inventory"
    }
    set f [open $out/clock_buffers.tsv w]
    puts $f "instance\tclock\tperiod\tloc\troute_status\tloads"
    foreach a {ntrpu_phy_reader_bufg ntrpu_phy_dma_bufg} port {clk dma_clk} clock {reader_clk dma_clk} period {5.0 4.0} {
        set c [exact_cell $a]
        set n [get_nets -of_objects [get_pins $a/O]]
        if {[get_clocks -of_objects [get_pins $a/O]] ne $clock || [get_property PERIOD [get_clocks $clock]]!=$period} {error "Changed boundary clock"}
        set input [get_nets -of_objects [get_pins $a/I]]
        if {[get_ports -of_objects $input] ne $port || [leaf_pins $input IN] ne "$a/I" || [llength [leaf_pins $input OUT]]} {error "Clock bypasses buffer"}
        set enable [get_cells -of_objects [leaf_pins [get_nets -of_objects [get_pins $a/CE]] OUT]]
        if {[llength $enable]!=1 || [get_property REF_NAME $enable] ne "VCC"} {error "Clock enable is not static"}
        set loads [leaf_pins $n IN]
        if {![llength $loads]} {error "Unloaded clock buffer"}
        foreach p $loads {if {[get_property REF_PIN_NAME $p] ni {C CLK}} {error "Clock reaches nonclock pin"}}
        puts $f "$a\t$clock\t$period\t[get_property LOC $c]\t[get_property ROUTE_STATUS $n]\t[llength $loads]"
    }
    close $f
    audit_boundary $manifest $out
    reset_scope $out 0
    report_route_status -file $out/route_status.rpt
    report_timing_summary -report_unconstrained -max_paths 10 -file $out/timing.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_drc -file $out/drc.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_cdc -from [get_clocks {reader_clk dma_clk}] -to [get_clocks {reader_clk dma_clk}] -details -all_checks_per_endpoint -no_waiver -file $out/clocked_cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored_exceptions.rpt
    report_bus_skew -max_paths 3 -file $out/bus_skew.rpt
    check_timing -verbose -file $out/check_timing.rpt
    write_xdc -constraints all $out/effective_constraints.xdc
    set f [open $out/metrics.tsv w]
    puts $f "metric\tvalue"
    puts $f "async_registers\t[llength [get_cells -hier -filter {ASYNC_REG == 1}]]"
    puts $f "blackboxes\t[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]"
    puts $f "latches\t[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]"
    foreach clock {reader_clk dma_clk} {
        set regs [all_registers -clock $clock]
        foreach {kind delay} {setup max hold min} {
            set p [get_timing_paths -from $regs -to $regs -delay_type $delay -max_paths 1]
            if {[llength $p]!=1} {error "Missing register timing"}
            puts $f "${clock}_${kind}\t[get_property SLACK $p]"
            report_timing -from $regs -to $regs -delay_type $delay -max_paths 10 -input_pins -file $out/${clock}_${kind}.rpt
        }
    }
    close $f
    topology_and_paths $root $out
}
proc boundary_run {root out} {
    file mkdir $out
    set_param general.maxThreads 2
    if {[info exists ::env(NTRPU_BOUNDARY_THREADS)]} {
        if {$::env(NTRPU_BOUNDARY_THREADS) ni {1 2 4}} {error "Unexpected CPU budget"}
        set_param general.maxThreads $::env(NTRPU_BOUNDARY_THREADS)
    }
    set f [open $root/tools/ntrpu_hbm_dma_rev2_physical.tcl];set text [read $f];close $f
    set boundary [string first "\nif \{\[catch \{physical" $text]
    if {$boundary<0} {error "Donor helper changed"}
    uplevel #0 [string range $text 0 [expr {$boundary-1}]]
    set f [open $root/evidence/mode.txt];set mode [string trim [read $f]];close $f
    if {$mode eq "prepare"} {
        open_checkpoint $root/evidence/controls_constrained.dcp
        create_cell -reference VCC ntrpu_phy_vcc_source
        create_net ntrpu_phy_vcc
        connect_net -net ntrpu_phy_vcc -objects [get_pins ntrpu_phy_vcc_source/P]
        insert_clock_buffer clk ntrpu_phy_reader_bufg
        insert_clock_buffer dma_clk ntrpu_phy_dma_bufg
        insert_boundary $out
        reset_scope $out 1
        write_checkpoint $out/prepared.dcp
        export_boundary $root $out $out/boundary.tsv
    } elseif {$mode eq "implement"} {
        open_checkpoint $root/evidence/prepared.dcp
        file copy $root/evidence/boundary.tsv $out/boundary.tsv
        opt_design
        write_checkpoint $out/optimized.dcp
        place_design
        write_checkpoint $out/placed.dcp
        report_timing_summary -report_unconstrained -file $out/placed_timing.rpt
        route_design -directive Default
        write_checkpoint $out/routed.dcp
        export_boundary $root $out $out/boundary.tsv
    } elseif {$mode eq "audit"} {
        open_checkpoint $root/evidence/routed.dcp
        file copy $root/evidence/boundary.tsv $out/boundary.tsv
        export_boundary $root $out $out/boundary.tsv
    } elseif {$mode eq "route"} {
        open_checkpoint $root/evidence/placed.dcp
        file copy $root/evidence/boundary.tsv $out/boundary.tsv
        route_design -directive RuntimeOptimized
        write_checkpoint $out/routed.dcp
        export_boundary $root $out $out/boundary.tsv
    } elseif {$mode eq "improve"} {
        open_checkpoint $root/evidence/routed.dcp
        file copy $root/evidence/boundary.tsv $out/boundary.tsv
        file mkdir $out/before_reset
        reset_scope $out/before_reset 0
        report_timing_summary -report_unconstrained -file $out/before_timing.rpt
        phys_opt_design -directive Explore
        write_checkpoint $out/optimized_routed.dcp
        route_design -directive RuntimeOptimized -preserve
        write_checkpoint $out/routed.dcp
        export_boundary $root $out $out/boundary.tsv
    } elseif {$mode eq "diagnose_reset"} {
        open_checkpoint $root/evidence/routed.dcp
        set f [open $out/reset_topology.tsv w]
        puts $f "kind\tpin\tref\tinit\tasync_reg"
        foreach p [leaf_pins [get_nets -of_objects [get_ports cold_reset_n]] IN] {
            set c [get_cells -of_objects $p]
            puts $f "direct\t$p\t[get_property REF_NAME $c]\t[get_property INIT $c]\t[get_property ASYNC_REG $c]"
        }
        foreach p [lsort [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]] {
            set c [get_cells -of_objects $p]
            puts $f "endpoint\t$p\t[get_property REF_NAME $c]\t[get_property INIT $c]\t[get_property ASYNC_REG $c]"
        }
        close $f
        report_route_status -file $out/route_status.rpt
        report_timing_summary -report_unconstrained -max_paths 10 -file $out/timing.rpt
        report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
        check_timing -verbose -file $out/check_timing.rpt
        write_xdc -constraints all $out/effective_constraints.xdc
    } else {error "Unknown mode: $mode"}
    puts NTRPU_BOUNDARY_EXPORTED
}
if {[catch {boundary_run $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
