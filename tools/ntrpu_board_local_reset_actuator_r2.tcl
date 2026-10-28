set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_local_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project local_reset $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_axil_cdc_recode ntrpu_axil_pair_drain ntrpu_axil_drain_monitor ntrpu_tagged_reset_mailbox ntrpu_reset_supervisor ntrpu_local_reset_actuator} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_host_axil_cdc
    set ip [get_ips ntrpu_host_axil_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
    generate_target all $ip
    report_property -all -file [file join $output cdc_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_local_reset_pair.sv]
    set_property top tb_ntrpu_local_reset_pair [get_filesets sim_1]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_local_reset_actuator_r2.sv]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set_property top tb_ntrpu_local_reset_actuator [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output local_reset.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output local.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "LOCAL_SUITE_PASS cases=8 hold_cycles=64 native_endpoint=0 complete_drain=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Local actuator bank failed"}
    set_property top tb_ntrpu_local_reset_pair [get_filesets sim_1]
    foreach period {5000 37000} {
        set_property generic [list DEVICE_PERIOD_PS=$period] [get_filesets sim_1]
        launch_simulation
        run all
        puts "NTRPU_PAIR_FINAL_TIME period=$period time=[current_time -s]"
        close_sim
        set log [file join $output local_reset.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output pair-${period}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "PAIR_SUITE_PASS cases=6 writes=4 reads=4 device_period_ps=$period native_pcie=0 complete_endpoint_drain=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Pair bank failed profile=$period"}
    }
    set clocks [file join $output clocks.xdc]
    set f [open $clocks {WRONLY CREAT EXCL}]
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    puts $f {create_clock -name host_clk -period 7.000 [get_ports host_clk]}
    puts $f {create_clock -name device_clk -period 37.000 [get_ports device_clk]}
    close $f
    add_files -fileset constrs_1 $clocks
    set_property top ntrpu_local_reset_actuator [get_filesets sources_1]
    synth_design -mode out_of_context -top ntrpu_local_reset_actuator -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    write_checkpoint [file join $output local_reset_synth.dcp]
    close_project
    puts NTRPU_LOCAL_RESET_EXPORTED
}
if {[catch {ntrpu_local_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
