# Private native-source integration. No example-board pins or blanket exceptions.
set_param general.maxThreads 2
proc exact_pin {name} {
    set p [get_pins -hierarchical -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin: $name"}
    return $p
}
proc integration_reports {out} {
    file mkdir $out
    report_clocks -file $out/clocks.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/guarded_cdc.rpt
    report_clock_interaction -file $out/clock_interaction.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_timing -from [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && IS_PRIMITIVE == 1}] -to [get_cells -hierarchical -filter {NAME =~ guarded/client/* && IS_PRIMITIVE == 1}] -delay_type min_max -max_paths 20 -file $out/guard_to_client.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_methodology -file $out/methodology.rpt
    report_drc -file $out/drc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/boundary.tsv {WRONLY CREAT EXCL}]
    puts $f "pin\tdirection\tnet\tclocks\tdrivers\tdriver_types\tendpoints"
    foreach name {guarded/clk guarded/mgmt_clk guarded/configuration_rst_n guarded/native_link_up guarded/native_axi_reset_n guarded/native_ctl_reset_n guarded/perst_n guarded/client_cold_rst_n guarded/client_link_up guarded/client_stop} {
        set p [exact_pin $name];set nets [get_nets -segments -of_objects $p]
        set drivers [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
        set types {};foreach d $drivers {lappend types [get_property REF_NAME [get_cells -of_objects $d]]}
        set endpoints [all_fanout -quiet -flat -endpoints_only -from $p]
        puts $f [join [list $name [get_property DIRECTION $p] [join $nets ,] [join [get_clocks -quiet -of_objects $p] ,] [join $drivers ,] [join $types ,] [join $endpoints ,]] "\t"]
    }
    close $f
    set f [open $out/guard_cells.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tasync_reg\tinit\tloc\tbel"
    foreach c [lsort [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && IS_PRIMITIVE == 1}]] {
        puts $f [join [list $c [get_property REF_NAME $c] [get_property ASYNC_REG $c] [get_property INIT $c] [get_property LOC $c] [get_property BEL $c]] "\t"]
    }
    close $f
}
if {[catch {
    if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {error "SOURCE EXAMPLE NEW_OUTPUT required"}
    lassign $argv source example out
    set source [file normalize $source];set example [file normalize $example];set out [file normalize $out]
    if {[file exists $out]} {error "New output required"}
    file mkdir $out
    create_project guarded_native $out/project -part xcvu47p-fsvh2892-2-e
    foreach name {ntrpu_bridge axi_bram_ctrl_0} {
        import_ip -files $example/ntrpu_bridge_ex.srcs/sources_1/ip/$name/$name.xci
        set ip [get_ips $name];set owned [file normalize [get_property IP_FILE $ip]]
        if {[string first "$out/" $owned]!=0} {error "IP import escaped private project"}
        set_property generate_synth_checkpoint false [get_files $owned]
        generate_target synthesis $ip
    }
    foreach {key value} {CONFIG.functional_mode AXI_Bridge CONFIG.axisten_freq 250 CONFIG.axi_addr_width 64 CONFIG.axi_data_width 512_bit} {
        if {[get_property $key [get_ips ntrpu_bridge]] ne $value} {error "Native IP profile changed: $key"}
    }
    foreach name {ntrpu_axi_bridge_guard_v2 ntrpu_axi_bridge_guarded_client ntrpu_axi_bridge_endpoint_client ntrpu_axi_bridge_writer ntrpu_axi_bridge_mapper ntrpu_xdma_local_master ntrpu_axi_bridge_guard_native} {
        read_verilog -sv $source/$name.sv
    }
    read_verilog $source/xilinx_axi_pcie_ep_app.v
    set_property top ntrpu_axi_bridge_guard_native [current_fileset]
    set f [open $out/private_clocks.xdc {WRONLY CREAT EXCL}]
    puts $f {create_clock -name pcie_refclk -period 10.000 [get_ports sys_clk_p]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk_in]}
    close $f
    read_xdc $out/private_clocks.xdc
    update_compile_order -fileset sources_1
    report_compile_order -used_in synthesis -file $out/compile_order.rpt
    report_ip_status -file $out/ip_status.rpt
    synth_design -top ntrpu_axi_bridge_guard_native -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
    write_checkpoint $out/native_synth.dcp
    write_verilog -mode funcsim $out/native_synth.v
    integration_reports $out/baseline
    if {[llength [get_cells -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes in native composition"}
    set g guarded/guard
    foreach {pin period} [list "$g/cold_m_reg\[0\]/C" 10.000 "$g/cold_a_reg\[0\]/C" 4.000] {
        set clocks [get_clocks -of_objects [exact_pin $pin]]
        if {[llength $clocks]!=1 || abs([get_property PERIOD $clocks]-$period)>0.001} {error "Native clock period unresolved or different: $pin $clocks"}
    }
    foreach {from to bound} [list request_reg/C {echo_a_reg[0]/D} 4.000 {echo_a_reg[2]/C} {echo_m_reg[0]/D} 10.000 armed_reg/C {armed_a_reg[0]/D} 4.000] {
        set_max_delay -datapath_only $bound -from [exact_pin $g/$from] -to [exact_pin $g/$to]
    }
    for {set n 0} {$n<4} {incr n} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [exact_pin [format {%s/availability[%d].release_m_reg[2]/C} $g $n]] -to [exact_pin [format {%s/availability[%d].release_a_reg[%d]/CLR} $g $n $i]]
        }
    }
    for {set i 0} {$i<3} {incr i} {
        set_false_path -from [exact_pin $g/fault_reg/C] -to [exact_pin [format {%s/stop_a_reg[%d]/PRE} $g $i]]
    }
    integration_reports $out/scoped
    write_checkpoint $out/native_scoped.dcp
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f "NATIVE_GUARDED_COMPOSITION_SYNTHESIZED_REVIEW_REQUIRED"
    close $f
    close_project
} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
puts NATIVE_GUARDED_COMPOSITION_SYNTHESIZED_REVIEW_REQUIRED
exit 0
