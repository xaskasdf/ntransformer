# NTRD, retained ownership, pool, cold supervisor and all four CDC instances.
# Endpoint PCIe and MIG PHY/controllers remain outside this OOC boundary.
# OOC internal timing only: parent I/O budgets and board placement are pending.
set script_dir [file dirname [file normalize [info script]]]
proc ntrpu_managed_host_synth {arguments} {
    global script_dir
    if {[llength $arguments] != 1 || [llength [get_projects -quiet]]} {
        error "Expected NEW_DIRECTORY in a fresh Vivado process"
    }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    set_param general.maxThreads 4
    create_project ingress [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_peer_multibuffer_fillstage ntrpu_rv_elastic ntrpu_peer_multibuffer_elastic ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits ntrpu_reset_supervisor ntrpu_dual_mig_cold_start ntrpu_dual_multibuffer_mig ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_managed ntrpu_pcie_fragment_dual_mig ntrpu_axil_cdc_recode ntrpu_pcie_fragment_dual_host_cold} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    foreach {name protocol width address} {ntrpu_ingress_axi_cdc AXI4 512 64 ntrpu_host_axil_cdc AXI4LITE 32 32 ntrpu_peer_memory_cdc AXI4 512 64} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
        set_property -dict $config $ip
        foreach {property value} $config {
            if {[get_property $property $ip] ne $value} {error "IP configuration differs: $property"}
        }
        generate_target all $ip
        report_property -all -file [file join $output ${name}_properties.rpt] $ip
    }
    report_ip_status -file [file join $output ip_status.rpt]
    # Disable only AMD's standalone placeholder clocks. Preserve scoped CDC
    # constraints and vendor waivers; add no false path or multicycle exception.
    set placeholders [get_files -all -quiet *_ooc.xdc]
    if {[llength $placeholders] != 3} {error "Expected three IP placeholder clock XDC"}
    set_property IS_ENABLED false $placeholders
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name host_clk -period 4.000 [get_ports host_clk]}
    puts $f {create_clock -name source_clk -period 5.000 [get_ports source_clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    puts $f {create_clock -name ui0_clk -period 5.000 [get_ports {ui_clk[0]}]}
    puts $f {create_clock -name ui1_clk -period 5.000 -waveform {0.731 3.231} [get_ports {ui_clk[1]}]}
    close $f
    read_xdc [file join $output clocks.xdc]
    set_property top ntrpu_pcie_fragment_dual_host_cold [get_filesets sources_1]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach file [get_files -compile_order sources -used_in synthesis] {puts $f $file}
    close $f
    set f [open [file join $output constraint_inputs.tsv] {WRONLY CREAT EXCL}]
    puts $f "path\tenabled\tused_in\tscoped_to_ref\tprocessing_order"
    foreach file [get_files -all -filter {FILE_TYPE == XDC}] {
        puts $f "$file\t[get_property IS_ENABLED $file]\t[get_property USED_IN $file]\t[get_property SCOPED_TO_REF $file]\t[get_property PROCESSING_ORDER $file]"
    }
    close $f
    # Expand optional-port diagnostics; do not alter severity or constraints.
    set_msg_config -id {Synth 8-7129} -limit 10000
    synth_design -mode out_of_context -top ntrpu_pcie_fragment_dual_host_cold -part xcvu5p-flva2104-2-i -generic {FIRST_COOKIE=64'hffffffff}
    if {[llength [get_clocks]] != 5} {error "Expected five clocks"}
    foreach {name period} {host_clk 4.0 source_clk 5.0 mgmt_clk 10.0 ui0_clk 5.0 ui1_clk 5.0} {
        if {[get_property PERIOD [get_clocks $name]] != $period} {error "Unexpected clock $name"}
    }
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Black box or latch"}
    write_checkpoint [file join $output managed_host_synth.dcp]
    write_verilog -mode funcsim [file join $output managed_host_funcsim.v]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# NTRD synthesis context\n\nHost clock 4 ns; source and both UI clocks 5 ns; management clock 10 ns. UI1 has 0.731 ns phase for the simulated fixture. All are independent parent ports, not generated board clocks."
    puts $f "Unchanged native-qualified NTRD/cold host wrapper, retained collector/control/bridge, real buffer pool and cold supervisor. Two host CDCs plus two memory CDC instances, from three AMD AXI Clock Converter 2.1 configurations. FIRST_COOKIE=0xffffffff matches the native suite."
    puts $f "PCIe endpoint and DDR4 controllers/PHY are outside this OOC boundary; their resources are not included. No compute is included. This is not the four-channel board or its pinout."
    puts $f "No input/output delays, parent clock locations, board pins or physical timing qualification. Only cold coordinated reset is qualified. External barriers remain trusted inputs; IRQ is source-domain."
    puts $f "No new false paths, multicycle exceptions, severity changes or user waivers. Generated vendor CDC constraints remain active. Reports require independent review before routing."
    close $f
    puts NTRPU_MANAGED_HOST_SYNTH_EXPORTED
    close_project
}
if {[catch {ntrpu_managed_host_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
