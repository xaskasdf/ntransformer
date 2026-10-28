# Small, bounded reproduction of port-origin CDC diagnostics, without RAM.
set root [file normalize [file join [file dirname [info script]] ..]]
proc snapshot {out name} {
    file mkdir $out/$name
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/$name/cdc.rpt
    check_timing -verbose -file $out/$name/check_timing.rpt
    write_xdc -constraints all $out/$name/constraints.xdc
    set f [open $out/$name/cells.tsv w]
    puts $f "name\tref"
    foreach c [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]] {puts $f "$c\t[get_property REF_NAME $c]"}
    close $f
    write_checkpoint $out/$name/design.dcp
}
proc probe {root out} {
    file mkdir $out
    set_param general.maxThreads 2
    if {[file exists $root/evidence/probe.dcp]} {
        open_checkpoint $root/evidence/probe.dcp
        foreach port {clk dma_clk} {
            set old [get_nets -of_objects [get_ports $port]]
            set a ${port}_buffer
            create_cell -reference BUFGCE $a
            create_net ${a}_input
            disconnect_net -net $old -objects [get_ports $port]
            connect_net -net ${a}_input -objects [concat [get_ports $port] [get_pins $a/I]]
            connect_net -net $old -objects [get_pins $a/O]
            connect_net -net [get_nets -of_objects [get_pins one/P]] -objects [get_pins $a/CE]
        }
        opt_design
        if {[file exists $root/evidence/opt_only]} {
            set f [open $out/reset_polarity.tsv w]
            puts $f "pin\tref\tclr_inverted\tasync_reg"
            foreach p [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_ports cold_reset_n]] -filter {DIRECTION == IN}] {
                set c [get_cells -of_objects $p]
                puts $f "$p\t[get_property REF_NAME $c]\t[get_property IS_CLR_INVERTED $c]\t[get_property ASYNC_REG $c]"
            }
            close $f
            snapshot $out optimized_reset
            puts NTRPU_RESET_LINK_PROBE_EXPORTED
            return
        }
        place_design
        route_design -directive Quick
        snapshot $out physical
        report_route_status -file $out/physical/route_status.rpt
        puts NTRPU_RESET_LINK_PROBE_EXPORTED
        return
    }
    read_verilog $root/fpga/experimental/hbm_dma_boundary/ntrpu_reset_link_probe.sv
    synth_design -top ntrpu_reset_link_probe -part xcvu47p-fsvh2892-2-e -mode out_of_context
    create_clock -name reader_clk -period 5 [get_ports clk]
    create_clock -name dma_clk -period 4 [get_ports dma_clk]
    set_input_delay -clock dma_clk -min 0.1 [get_ports dma_link_up]
    set_input_delay -clock dma_clk -max 1.0 [get_ports dma_link_up]
    set_output_delay -clock reader_clk -min -0.1 [get_ports {reader_ready link_seen}]
    set_output_delay -clock reader_clk -max 1.0 [get_ports {reader_ready link_seen}]
    set_output_delay -clock dma_clk -min -0.1 [get_ports dma_ready]
    set_output_delay -clock dma_clk -max 1.0 [get_ports dma_ready]
    set cold [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]
    if {[llength $cold]!=8} {error "Unexpected reset topology: $cold"}
    foreach p $cold {if {[get_property REF_PIN_NAME $p] ne "CLR"} {error "Unexpected reset endpoint"}}
    set_false_path -from [get_ports cold_reset_n] -to $cold
    write_checkpoint $out/common.dcp
    set ld [get_pins link_s1_reg/D]
    set_max_delay -datapath_only 4 -from [get_ports dma_link_up] -to $ld
    snapshot $out port_origin
    # Only link-source representation changes. Raw reset is retained unchanged.
    close_design
    open_checkpoint $out/common.dcp
    set ld [get_pins link_s1_reg/D]
    create_cell -reference FDRE launch
    create_cell -reference VCC one
    create_cell -reference GND zero
    create_net one;create_net zero;create_net launch_input
    connect_net -net one -objects [get_pins {one/P launch/CE}]
    connect_net -net zero -objects [get_pins {zero/G launch/R}]
    set_property DONT_TOUCH true [get_cells launch]
    set old [get_nets -of_objects [get_ports dma_link_up]]
    disconnect_net -net $old -objects [get_ports dma_link_up]
    connect_net -net launch_input -objects [concat [get_ports dma_link_up] [get_pins launch/D]]
    connect_net -net $old -objects [get_pins launch/Q]
    connect_net -net [get_nets -of_objects [get_ports dma_clk]] -objects [get_pins launch/C]
    set_max_delay -datapath_only 4 -from [get_cells launch] -to $ld
    snapshot $out register_origin
    set_false_path -from [get_ports dma_link_up] -to [get_pins launch/D]
    snapshot $out outer_cut
    puts NTRPU_RESET_LINK_PROBE_EXPORTED
}
if {[catch {probe $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
