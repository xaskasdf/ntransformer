# Isolated physical experiment on the frozen revision-2 checkpoint.
# Interface budgets are a provisional same-chip contract, not board pin timing.
set root [file normalize [file join [file dirname [info script]] ..]]
proc exact_cell {name} {
    set c [get_cells -hier -filter [list NAME == $name]]
    if {[llength $c]!=1} {error "Missing/duplicated cell: $name"}
    return $c
}
proc insert_clock_buffer {port instance} {
    set p [get_ports $port]
    set net [get_nets -of_objects $p]
    if {[llength $net]!=1} {error "Ambiguous clock net"}
    set before [lsort [get_pins -leaf -of_objects [get_nets -segments $net] -filter {DIRECTION == IN}]]
    create_cell -reference BUFGCE $instance
    create_net ${instance}_input
    disconnect_net -net $net -objects $p
    connect_net -net ${instance}_input -objects [concat $p [get_pins $instance/I]]
    connect_net -net $net -objects [get_pins $instance/O]
    connect_net -net ntrpu_phy_vcc -objects [get_pins $instance/CE]
    set after [lsort [get_pins -leaf -of_objects [get_nets -segments $net] -filter {DIRECTION == IN}]]
    if {$before ne $after} {error "Clock consumers changed during insertion"}
}
proc interface_budget {out} {
    set f [open $out/interface.tsv w]
    puts $f "port\tdirection\tclock\tmin_ns\tmax_ns"
    foreach p [lsort [get_ports]] {
        set n [get_property NAME $p]
        if {$n in {clk dma_clk cold_reset_n}} {continue}
        set direction [get_property DIRECTION $p]
        set base [lindex [split $n {[}] 0]
        if {$direction eq "IN"} {
            set dc [expr {$base in {dma_link_up dma_run dma_error dma_done dsc_ready axis_ready}}]
            set clock [expr {$dc ? "dma_clk" : "reader_clk"}]
            set_input_delay -clock [get_clocks $clock] -min 0.100 $p
            set_input_delay -clock [get_clocks $clock] -max 1.000 $p
            puts $f "$n\t$direction\t$clock\t0.100\t1.000"
        } elseif {$direction eq "OUT"} {
            set dc [expr {$base in {dma_rst_n dma_fault dsc_load axis_valid axis_last dsc_src dsc_dst axis_keep dsc_len dsc_ctl axis_data}}]
            set clock [expr {$dc ? "dma_clk" : "reader_clk"}]
            set_output_delay -clock [get_clocks $clock] -min -0.100 $p
            set_output_delay -clock [get_clocks $clock] -max 1.000 $p
            puts $f "$n\t$direction\t$clock\t-0.100\t1.000"
        } else {error "Unexpected interface direction"}
    }
    close $f
    # Cold assertion/release enters only the reset-startup synchronizers.
    # Internal synchronous release paths remain timed. This does not qualify
    # external reset pulse width, coordinated isolation or PCIe quiescence.
    set reset_input_loads [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_ports cold_reset_n]] -filter {DIRECTION == IN}]
    if {[llength $reset_input_loads]!=1} {error "Unexpected reset polarity logic"}
    set inverter [get_cells -of_objects $reset_input_loads]
    if {[get_property REF_NAME $inverter] ne "LUT1" || [get_property INIT $inverter] ne "2'h1"} {error "Reset is not a single inversion"}
    set reset_loads [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]
    if {[llength $reset_loads]!=10} {error "Unexpected cold-reset fanout"}
    foreach p $reset_loads {
        if {[get_property REF_PIN_NAME $p] ne "CLR" || ![get_property ASYNC_REG [get_cells -of_objects $p]]} {
            error "Cold reset reaches unexpected consumer: $p"
        }
    }
    set_false_path -from [get_ports cold_reset_n] -to $reset_loads
    set f [open $out/cold_reset_endpoints.txt w]
    puts $f [join [lsort $reset_loads] \n]
    close $f
    # Link status is launched by the DMA interface and has a dedicated reader
    # synchronizer. Bound only its first stage; direct DMA consumers stay timed.
    set link_d [get_pins -of_objects [exact_cell link_s1_reg] -filter {REF_PIN_NAME == D}]
    set_max_delay -datapath_only 4.000 -from [get_ports dma_link_up] -to $link_d
}
proc topology_and_paths {root out} {
    # Import the already reviewed topology procedures, not their entry point.
    set f [open $root/tools/ntrpu_hbm_dma_rev2_capture_review.tcl];set text [read $f];close $f
    set boundary [string first {proc review {root out}} $text]
    if {$boundary<0} {error "Topology helper boundary changed"}
    uplevel #0 [string range $text 0 [expr {$boundary-1}]]
    global data_bindings data_groups
    set data_groups [dict create]
    set data_bindings [open $out/data_bindings.tsv w]
    puts $data_bindings "fifo\tram_cells\tbits\tsource_clock\tdest_clock\tunused_outputs"
    foreach lane {0 1} {
        foreach {fifo width} {commands 143 payloads 514 completions 2} {
            set sc [expr {$fifo eq "completions" ? "dma_clk" : "reader_clk"}]
            set dc [expr {$fifo eq "completions" ? "reader_clk" : "dma_clk"}]
            bind_capture $lane $fifo $width $sc $dc
        }
    }
    close $data_bindings
    set f [open $out/data_paths.tsv w]
    puts $f "fifo\tstartpoint\tendpoint\tslack"
    dict for {name group} $data_groups {
        lassign $group memories dpins width
        set paths [get_timing_paths -from $memories -to $dpins -max_paths $width -nworst 1]
        if {[llength $paths]!=$width} {error "Incomplete capture coverage: $name"}
        foreach p $paths {puts $f "$name\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]\t[get_property SLACK $p]"}
        report_timing -from $memories -to $dpins -max_paths 3 -input_pins -file $out/${name}_capture.rpt
    }
    close $f
    set f [open $root/evidence/bindings.tsv];set lines [split [read $f] \n];close $f
    set o [open $out/control_paths.tsv w]
    puts $o "group\tsource\tfirst\tsecond\tsource_clock\tdest_clock\tgray\tslack"
    foreach line [lrange $lines 1 end] {
        if {[string trim $line] eq ""} {continue}
        lassign [split [string trim $line] \t] group source first second sc dc gray
        set s [exact_cell $source];set a [exact_cell $first];set b [exact_cell $second]
        foreach cell [list $s $a $b] clock [list $sc $dc $dc] {
            if {[get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]] ne $clock} {error "Control clock changed"}
        }
        foreach cell [list $a $b] {if {![get_property ASYNC_REG $cell]} {error "Missing synchronizer property"}}
        set dp [get_pins -of_objects $a -filter {REF_PIN_NAME == D}]
        set driver [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $dp] -filter {DIRECTION == OUT}]]
        if {$driver ne $s} {error "First stage driver changed: $first"}
        set qp [get_pins -of_objects $a -filter {REF_PIN_NAME == Q}]
        set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects $qp] -filter {DIRECTION == IN}]
        if {[llength $loads]!=1 || [get_cells -of_objects $loads] ne $b || [get_property REF_PIN_NAME $loads] ne "D"} {error "Synchronizer fanout changed"}
        set p [get_timing_paths -from $s -to $dp -max_paths 1]
        if {[llength $p]!=1} {error "Missing control timing path"}
        puts $o "$group\t$source\t$first\t$second\t$sc\t$dc\t$gray\t[get_property SLACK $p]"
    }
    close $o
}
proc physical {root out} {
    file mkdir $out
    set_param general.maxThreads 2
    open_checkpoint $root/evidence/controls_constrained.dcp
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong device"}
    foreach {clock period} {reader_clk 5.0 dma_clk 4.0} {
        if {[get_property PERIOD [get_clocks $clock]]!=$period} {error "Wrong clock period"}
    }
    create_cell -reference VCC ntrpu_phy_vcc_source
    create_net ntrpu_phy_vcc
    connect_net -net ntrpu_phy_vcc -objects [get_pins ntrpu_phy_vcc_source/P]
    insert_clock_buffer clk ntrpu_phy_reader_bufg
    insert_clock_buffer dma_clk ntrpu_phy_dma_bufg
    interface_budget $out
    write_checkpoint $out/prepared.dcp
    write_xdc -constraints all $out/prepared.xdc
    opt_design
    place_design -directive Explore
    phys_opt_design -directive Explore
    write_checkpoint $out/placed.dcp
    report_timing_summary -report_unconstrained -file $out/placed_timing.rpt
    route_design -directive Explore
    write_checkpoint $out/routed.dcp
    # Save routing/timing even if a strict topology invariant needs re-review.
    report_route_status -file $out/route_status.rpt
    report_timing_summary -report_unconstrained -max_paths 10 -file $out/timing.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_drc -file $out/drc.rpt
    report_methodology -file $out/methodology.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored_exceptions.rpt
    report_bus_skew -max_paths 3 -file $out/bus_skew.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_clock_interaction -file $out/clock_interaction.rpt
    write_xdc -constraints all $out/effective_constraints.xdc
    set f [open $out/metrics.tsv w]
    foreach kind {PLACED_FULLY PARTIALLY_ROUTED ROUTED_FULLY ERRORS_IN_ROUTES} {puts $f "$kind\t[get_property $kind [current_design]]"}
    foreach clock {reader_clk dma_clk} {
        foreach {kind delay} {setup max hold min} {
            set paths [get_timing_paths -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 1]
            puts $f "${clock}_${kind}\t[get_property SLACK $paths]"
            report_timing -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 10 -input_pins -file $out/${clock}_${kind}.rpt
        }
    }
    puts $f "async_registers\t[llength [get_cells -hier -filter {ASYNC_REG == 1}]]"
    close $f
    set f [open $out/clock_buffers.tsv w]
    puts $f "instance\tref\tloc\tclock\toutput_net\troute_status\tloads\tinput_port\tce_driver"
    foreach inst {ntrpu_phy_reader_bufg ntrpu_phy_dma_bufg} {
        set c [exact_cell $inst]
        set op [get_pins $inst/O];set nets [get_nets -segments -of_objects $op]
        set loads [get_pins -leaf -of_objects $nets -filter {DIRECTION == IN}]
        set ip [get_ports -of_objects [get_nets -segments -of_objects [get_pins $inst/I]]]
        set ce [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $inst/CE]] -filter {DIRECTION == OUT}]]
        puts $f "$inst\t[get_property REF_NAME $c]\t[get_property LOC $c]\t[get_clocks -of_objects $op]\t[get_nets -of_objects $op]\t[get_property ROUTE_STATUS [get_nets -of_objects $op]]\t[llength $loads]\t$ip\t[get_property REF_NAME $ce]"
    }
    close $f
    topology_and_paths $root $out
    puts NTRPU_BRIDGE_REV2_PHYSICAL_EXPORTED
}
if {[catch {physical $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
