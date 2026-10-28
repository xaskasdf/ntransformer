set root [file normalize [file join [file dirname [info script]] ..]]
set output [file normalize [lindex $argv 0]]
if {[catch {
    set threads [lindex $argv 1]
    if {$threads eq ""} {set threads 2}
    set_param general.maxThreads $threads
    set f [open [file join $root tools ntrpu_read_return.tcl]];set text [read $f];close $f
    set end [string first {if {[llength $argv]!=3}} $text]
    if {$end<0} {error "Read configuration entry boundary changed"}
    eval [string range $text 0 [expr {$end-1}]]
    prepare_read_return $root $output
    foreach n {ntrpu_peer_read_pkg ntrpu_peer_read_lane ntrpu_peer_read_pair} {add_files [file join $root fpga experimental ${n}.sv]}
    update_compile_order -fileset sources_1
    add_files [file join $output ntrpu_peer_read_physical.sv]
    set_property top ntrpu_peer_read_physical [get_filesets sources_1]
    update_compile_order -fileset sources_1
    report_compile_order -fileset sources_1 -used_in synthesis -file [file join $output compile_order.rpt]
    set f [open [file join $output clocks.xdc] w]
    puts $f {create_clock -name transport_clk -period 5.000 [get_ports clk]}
    puts $f {create_clock -name memory_clk -period 2.222 [get_ports mem_clk]}
    puts $f {create_clock -name apb_status_clk -period 10.000}
    puts $f {set_input_delay -clock apb_status_clk -min 0.000 [get_ports init_complete_async*]}
    puts $f {set_input_delay -clock apb_status_clk -max 0.000 [get_ports init_complete_async*]}
    puts $f {set_clock_uncertainty -setup 0.100 [get_clocks {transport_clk memory_clk}]}
    puts $f {set_clock_uncertainty -hold 0.050 [get_clocks {transport_clk memory_clk}]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    synth_design -top ntrpu_peer_read_physical -mode out_of_context -part xcvu47p-fsvh2892-2-e -flatten_hierarchy rebuilt
    read_xdc [file join $root fpga constraints ntrpu_hbm_startup_cdc.xdc]
    if {[llength $ntrpu_status_internal_d]!=3 || [llength $ntrpu_status_apb_d]!=2} {error "Status constraints missed endpoints"}
    foreach c [get_cells -hier -regexp {(^|.*/)startup/(ready_meta_reg\[[0-2]\]|fault_meta_reg\[[0-1]\])}] {
        if {![get_property ASYNC_REG $c]} {error "Missing ASYNC_REG on $c"}
    }
    report_utilization -file [file join $output synthesis_utilization.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output synthesis_cdc.rpt]
    write_checkpoint [file join $output synthesized.dcp]
    opt_design
    write_checkpoint [file join $output optimized.dcp]
    place_design
    write_checkpoint [file join $output placed.dcp]
    phys_opt_design -directive Explore
    route_design -directive Explore
    write_checkpoint [file join $output routed.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_route_status -file [file join $output route_status.rpt]
    report_drc -file [file join $output drc.rpt]
    write_verilog -mode funcsim [file join $output routed.v]
    write_xdc -constraints all [file join $output constraints.xdc]
    set f [open [file join $output ready_fanin.tsv] w]
    puts $f "output\tstartpoint\tclass"
    set ready_ports [get_ports -regexp {h_rready\[[01]\]}]
    if {[llength $ready_ports]!=2} {error "Missing native READY ports"}
    foreach port $ready_ports {
        set starts [all_fanin -flat -startpoints_only -to $port]
        if {![llength $starts]} {error "Missing native READY fanin"}
        foreach start $starts {puts $f "$port\t$start\t[get_property CLASS $start]"}
    }
    close $f
    set f [open [file join $output summary.tsv] w]
    foreach mode {max min} {
        puts $f "${mode}_slack_ns\t[get_property SLACK [get_timing_paths -delay_type $mode -max_paths 1]]"
    }
    foreach name {transport_clk memory_clk} {
        set c [get_clocks $name]
        puts $f "period_$name\t[get_property PERIOD $c]"
        foreach mode {max min} {
            puts $f "${mode}_$name\t[get_property SLACK [get_timing_paths -from $c -to $c -delay_type $mode -max_paths 1]]"
            report_timing -from $c -to $c -delay_type $mode -max_paths 10 -file [file join $output ${name}_${mode}.rpt]
        }
    }
    puts $f "startup_async_registers\t[llength [get_cells -hier -regexp {(^|.*/)startup/.*} -filter {ASYNC_REG == TRUE}]]"
    puts $f "internal_status_pins\t[llength $ntrpu_status_internal_d]"
    puts $f "external_status_pins\t[llength $ntrpu_status_apb_d]"
    puts $f "latches\t[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]"
    puts $f "blackboxes\t[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts PEER_READ_PHYSICAL_EXPORTED
exit 0
