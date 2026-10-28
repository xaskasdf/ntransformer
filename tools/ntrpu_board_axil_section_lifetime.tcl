set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_shutdown_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project shutdown $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_axil_cdc_recode ntrpu_axil_pair_drain ntrpu_axil_drain_monitor ntrpu_tagged_reset_mailbox ntrpu_reset_supervisor ntrpu_local_reset_actuator ntrpu_axil_retained_admission_r2 ntrpu_axil_fence_certificate_r4 ntrpu_axil_fence_section_r2} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_host_axil_cdc
    set ip [get_ips ntrpu_host_axil_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
    generate_target all $ip
    report_property -all -file [file join $output cdc_properties.rpt] $ip
    create_ip -vlnv xilinx.com:ip:dfx_axi_shutdown_manager:1.0 -module_name ntrpu_host_axil_shutdown
    set shutdown [get_ips ntrpu_host_axil_shutdown]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $shutdown]]
    set_property -dict {CONFIG.RESET_ACTIVE_LEVEL 0 CONFIG.RP_IS_MASTER false CONFIG.CTRL_INTERFACE_TYPE 0 CONFIG.DP_PROTOCOL AXI4LITE CONFIG.DP_AXI_ADDR_WIDTH 32 CONFIG.DP_AXI_DATA_WIDTH 32 CONFIG.DP_AXI_RESP 0} $shutdown
    generate_target all $shutdown
    report_property -all -file [file join $output shutdown_properties.rpt] $shutdown
    report_ip_status -file [file join $output ip_status.rpt]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axil_section_lifetime.sv]
    set_property top tb_ntrpu_axil_section_lifetime [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach stop {0 1} {
      foreach period {5000 37000} {
        set_property generic [list DEVICE_PERIOD_PS=$period STOP_SOURCE=$stop] [get_filesets sim_1]
        launch_simulation
        run all
        puts "NTRPU_SHUTDOWN_FINAL_TIME period=$period time=[current_time -s]"
        close_sim
        set log [file join $output shutdown.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output shutdown-${stop}-${period}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "LIFE_SUITE_PASS cases=4 device_period_ps=$period stop_source=$stop native_endpoint=0 complete_dma_drain=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Shutdown bank failed profile=$period"}
    }
    }
    close_project
    puts NTRPU_SHUTDOWN_LIFETIME_EXPORTED
}
if {[catch {ntrpu_shutdown_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
