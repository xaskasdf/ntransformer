set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_shutdown_synth {arguments} {
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project shutdown_ip $output -part xcvu5p-flva2104-2-i
    create_ip -vlnv xilinx.com:ip:dfx_axi_shutdown_manager:1.0 -module_name ntrpu_host_axil_shutdown
    set ip [get_ips ntrpu_host_axil_shutdown]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    # The retained manager is upstream of the resettable CDC. Mode 0 means
    # SLVERR (not AXI RESP encoding); RP_IS_MASTER=false preserves the host side.
    set_property -dict {CONFIG.RESET_ACTIVE_LEVEL 0 CONFIG.RP_IS_MASTER false CONFIG.CTRL_INTERFACE_TYPE 0 CONFIG.DP_PROTOCOL AXI4LITE CONFIG.DP_AXI_ADDR_WIDTH 32 CONFIG.DP_AXI_DATA_WIDTH 32 CONFIG.DP_AXI_RESP 0} $ip
    report_property -all -file [file join $output shutdown_properties.rpt] $ip
    generate_target all $ip
    report_ip_status -file [file join $output ip_status.rpt]
    set clocks [file join $output clocks.xdc]
    set f [open $clocks {WRONLY CREAT EXCL}]
    puts $f {create_clock -name host_clk -period 7.000 [get_ports clk]}
    close $f
    add_files -fileset constrs_1 $clocks
    set_property top ntrpu_host_axil_shutdown [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -mode out_of_context -top ntrpu_host_axil_shutdown -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    write_checkpoint [file join $output shutdown_synth.dcp]
    close_project
    puts NTRPU_SHUTDOWN_SYNTH_EXPORTED
}
if {[catch {ntrpu_shutdown_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
