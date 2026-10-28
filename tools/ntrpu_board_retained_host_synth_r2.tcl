set root [file normalize [file join [file dirname [info script]] ..]]
proc retained_host_synth {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project retained_host [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_axi4_drain_monitor_r2 ntrpu_axi4_pair_drain_r2 ntrpu_axi4_response_guard_r2 ntrpu_axi4_retained_admission_r2 ntrpu_axi4_section_certificate_r2 ntrpu_axil_cdc_recode ntrpu_axil_drain_monitor ntrpu_axil_fence_certificate_r5 ntrpu_axil_fence_section_r3 ntrpu_axil_pair_drain ntrpu_axil_response_guard ntrpu_axil_retained_admission_r2 ntrpu_drain_certificate_join ntrpu_fragment_pool_bridge ntrpu_local_reply_certificate_r2 ntrpu_owned_drain_coordinator ntrpu_ownership_certificate ntrpu_pcie_fragment_certified_owner ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_coordinated ntrpu_pcie_fragment_ingress ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_retained_host ntrpu_reset_supervisor ntrpu_retained_axi4_path ntrpu_retained_axil_path ntrpu_tagged_reset_mailbox} {
        add_files [file join $root fpga rtl $module.sv]
    }
    foreach {name protocol width address id} {ntrpu_axi4_cdc_probe AXI4 512 64 4 ntrpu_host_axil_cdc AXI4LITE 32 32 0} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set_property -dict [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $ip
        if {$protocol eq "AXI4"} {set_property CONFIG.ID_WIDTH $id $ip}
        generate_target all $ip
        report_property -all -file [file join $output ${name}_properties.rpt] $ip
    }
    create_ip -vlnv xilinx.com:ip:dfx_axi_shutdown_manager:1.0 -module_name ntrpu_host_axil_shutdown
    set ip [get_ips ntrpu_host_axil_shutdown]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.RESET_ACTIVE_LEVEL 0 CONFIG.RP_IS_MASTER false CONFIG.CTRL_INTERFACE_TYPE 0 CONFIG.DP_PROTOCOL AXI4LITE CONFIG.DP_AXI_ADDR_WIDTH 32 CONFIG.DP_AXI_DATA_WIDTH 32 CONFIG.DP_AXI_RESP 0} $ip
    generate_target all $ip
    report_property -all -file [file join $output shutdown_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name source_clk -period 5.000 [get_ports source_clk]}
    puts $f {create_clock -name host_clk -period 7.000 [get_ports host_clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_pcie_fragment_retained_host [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -mode out_of_context -top ntrpu_pcie_fragment_retained_host -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output retained_host_utilization.rpt]
    report_timing_summary -file [file join $output retained_host_timing.rpt]
    check_timing -verbose -file [file join $output retained_host_check_timing.rpt]
    report_cdc -details -file [file join $output retained_host_cdc.rpt]
    report_exceptions -file [file join $output retained_host_exceptions.rpt]
    report_clock_interaction -file [file join $output retained_host_clock_interaction.rpt]
    set f [open [file join $output retained_host_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    # Actual public ready boundaries: no fictitious board I/O delays.
    report_timing -max_paths 20 -path_type full_clock_expanded -to [get_ports {s_axi_awready s_axi_wready s_axil_awready s_axil_wready s_axil_arready}] -file [file join $output ready_outputs.rpt]
    report_timing -max_paths 20 -path_type full_clock_expanded -from [get_ports {s_axi_bready s_axil_bready s_axil_rready}] -file [file join $output ready_inputs.rpt]
    set f [open [file join $output source_files.tsv] {WRONLY CREAT EXCL}]
    puts $f "path\ttype\tused_in_synthesis\tprocessing_order"
    foreach source [get_files -all] {
        puts $f "$source\t[get_property FILE_TYPE $source]\t[get_property USED_IN_SYNTHESIS $source]\t[get_property -quiet PROCESSING_ORDER $source]"
    }
    close $f
    write_checkpoint [file join $output retained_host_synth.dcp]
    write_verilog -mode funcsim [file join $output retained_host_funcsim.v]
    close_project
    puts NTRPU_RETAINED_HOST_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_host_synth $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
