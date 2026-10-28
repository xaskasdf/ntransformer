# Physical composition only; no board queue, endpoint or memory controller.
set script_dir [file dirname [file normalize [info script]]]
proc composition {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set part xcvu5p-flva2104-2-i
    exec python3 -B [file join $script_dir prepare_ntrpu_composition_fixture.py] [file join $output fixture]
    create_project composition [file join $output project] -part $part
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_peer_memory_cdc
    set ip [get_ips ntrpu_peer_memory_cdc]
    set config {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3}
    set_property -dict $config $ip
    foreach {key value} $config {
        if {[get_property $key $ip] ne $value} {error "CDC configuration mismatch $key"}
    }
    set_property GENERATE_SYNTH_CHECKPOINT false [get_files *ntrpu_peer_memory_cdc.xci]
    generate_target all $ip
    report_property -all $ip -file [file join $output cdc_configured.rpt]
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded
        ntrpu_peer_multibuffer_fillstage ntrpu_rv_elastic ntrpu_peer_multibuffer_elastic
        ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    add_files [file join $output fixture fixture.sv]
    set xdc [file join $output fixture.xdc]
    set f [open $xdc w]
    puts $f {create_clock -period 5.000 -name source_clk [get_ports clk]}
    puts $f {create_clock -period 4.000 -name memory0_clk [get_ports {mem_clk[0]}]}
    puts $f {create_clock -period 4.300 -name memory1_clk [get_ports {mem_clk[1]}]}
    puts $f {set_clock_uncertainty -setup 0.100 [get_clocks]}
    puts $f {set_clock_uncertainty -hold 0.050 [get_clocks]}
    close $f
    add_files -fileset constrs_1 $xdc
    set_property top ntrpu_composition_fixture [get_filesets sources_1]
    update_compile_order -fileset sources_1
    report_compile_order -constraints -file [file join $output constraint_order.rpt]
    synth_design -top ntrpu_composition_fixture -part $part -mode out_of_context -flatten_hierarchy rebuilt
    if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX==1}]]} {error "Unexpected black boxes"}
    write_checkpoint [file join $output synthesized.dcp]
    report_utilization -hierarchical -file [file join $output synth_utilization.rpt]
    opt_design
    place_design -directive Default
    write_checkpoint [file join $output placed.dcp]
    phys_opt_design
    route_design -directive RuntimeOptimized
    write_checkpoint [file join $output routed.dcp]
    write_verilog -cell core -mode funcsim -rename_top ntrpu_composition_core [file join $output routed_core_funcsim.v]
    write_xdc [file join $output implemented.xdc]
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_utilization -cells [get_cells core] -file [file join $output core_utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_timing -delay_type max -max_paths 20 -file [file join $output setup_paths.rpt]
    report_timing -delay_type min -max_paths 20 -file [file join $output hold_paths.rpt]
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc_unwaived.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    report_route_status -file [file join $output route_status.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_waivers -file [file join $output waivers.rpt]
    set f [open [file join $output metrics.txt] w]
    foreach {label clock} {SOURCE source_clk MEMORY0 memory0_clk MEMORY1 memory1_clk} {
        puts $f "${label}_PERIOD_NS=[get_property PERIOD [get_clocks $clock]]"
        foreach {kind delay} {SETUP max HOLD min} {
            set paths [get_timing_paths -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 1]
            if {![llength $paths]} {error "No $kind path on $clock"}
            puts $f "${label}_${kind}_NS=[get_property SLACK $paths]"
        }
    }
    puts $f "CLOCKS=[llength [get_clocks]]"
    puts $f "BUFGS=[llength [get_cells -hier -filter {REF_NAME==BUFGCE}]]"
    puts $f "DRC_ERRORS=[llength [get_drc_violations -quiet -filter {SEVERITY==Error}]]"
    puts $f "BLACKBOXES=[llength [get_cells -hier -quiet -filter {IS_BLACKBOX==1}]]"
    close $f
    close_project
    puts NTRPU_COMPOSITION_PHYSICAL_EXPORTED
}
if {[catch {composition $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
