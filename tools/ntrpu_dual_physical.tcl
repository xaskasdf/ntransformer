# Isolated OOC receiver + cold supervisor, bounded at the two MIG AXI interfaces.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir ntrpu_dual_physical_reports.tcl]
proc composition {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set part xcvu5p-flva2104-2-i
    exec python3 -B [file join $script_dir prepare_ntrpu_dual_physical_fixture.py] [file join $output fixture]
    create_project dual_physical [file join $output project] -part $part
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
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
        ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits ntrpu_reset_supervisor ntrpu_dual_mig_cold_start ntrpu_dual_multibuffer_mig} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    add_files [file join $output fixture fixture.sv]
    set xdc [file join $output fixture.xdc]
    set f [open $xdc w]
    puts $f {create_clock -period 5.000 -name source_clk [get_ports source_clk]}
    puts $f {create_clock -period 5.000 -name memory0_clk [get_ports {ui_clk[0]}]}
    puts $f {create_clock -period 5.000 -name memory1_clk [get_ports {ui_clk[1]}]}
    puts $f {create_clock -period 10.000 -name mgmt_clk [get_ports mgmt_clk]}
    puts $f {set_clock_uncertainty -setup 0.100 [get_clocks]}
    puts $f {set_clock_uncertainty -hold 0.050 [get_clocks]}
    close $f
    add_files -fileset constrs_1 $xdc
    set_property top ntrpu_dual_physical_fixture [get_filesets sources_1]
    update_compile_order -fileset sources_1
    report_compile_order -constraints -file [file join $output constraint_order.rpt]
    synth_design -top ntrpu_dual_physical_fixture -part $part -mode out_of_context -flatten_hierarchy rebuilt
    if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX==1}]]} {error "Unexpected black boxes"}
    file mkdir [file join $output synthesized_audit]
    dual_release_chains [file join $output synthesized_audit] 1
    write_checkpoint [file join $output synthesized.dcp]
    report_utilization -hierarchical -file [file join $output synth_utilization.rpt]
    opt_design
    place_design -directive Default
    write_checkpoint [file join $output placed.dcp]
    phys_opt_design
    route_design -directive RuntimeOptimized
    write_checkpoint [file join $output routed.dcp]
    write_verilog -cell core -mode funcsim -rename_top ntrpu_dual_routed_core [file join $output routed_core_funcsim.v]
    write_verilog -cell core/startup -mode funcsim -rename_top ntrpu_dual_routed_startup [file join $output routed_startup_funcsim.v]
    dual_physical_reports $output
    close_project
    puts NTRPU_DUAL_PHYSICAL_EXPORTED
}
if {[catch {composition $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
