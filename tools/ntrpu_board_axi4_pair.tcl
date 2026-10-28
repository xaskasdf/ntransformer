set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc axi4_pair_bank {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project pair $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_axi4_pair_drain ntrpu_axi4_drain_monitor ntrpu_tagged_reset_mailbox} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name.sv]
    }
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_axi4_cdc_probe
    set ip [get_ips ntrpu_axi4_cdc_probe]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
    generate_target all $ip
    report_property -all -file [file join $output cdc_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_pair_drain.sv]
    set_property top tb_ntrpu_axi4_pair_drain [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach period {5000 37000} {
        set_property generic [list DEVICE_PERIOD_PS=$period] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output pair.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output pair-$period.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "AXI4_PAIR_SUITE_PASS device_period_ps=$period writes=24 reads=24 proofs=3 native_endpoint=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "AXI4 pair failed profile=$period"}
    }
    close_project
    puts NTRPU_AXI4_PAIR_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_pair_bank [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
