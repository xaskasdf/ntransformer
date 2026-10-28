set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc axi4_section_bank {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project pair $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_axi4_section_certificate_r2 ntrpu_axi4_retained_admission_r2 ntrpu_axi4_response_guard_r2 ntrpu_axi4_pair_drain_r2 ntrpu_axi4_drain_monitor_r2 ntrpu_tagged_reset_mailbox} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name.sv]
    }
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_axi4_cdc_probe
    set ip [get_ips ntrpu_axi4_cdc_probe]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
    generate_target all $ip
    report_property -all -file [file join $output cdc_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_section_forward.sv]
    set_property top tb_ntrpu_axi4_section_forward [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach period {37000 5000} {
        foreach kind {0 1 2 3} {
            set key forward-$kind-$period
            set_property generic [list DEVICE_PERIOD_PS=$period CASE_KIND=$kind] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set simdir [file join $output pair.sim sim_1 behav xsim]
            set log [file join $simdir simulate.log]
            file copy $log [file join $output $key.log]
            foreach leaf {elaborate.log xelab.log xvlog.log compile.log} {
                if {[file isfile [file join $simdir $leaf]]} {file copy [file join $simdir $leaf] [file join $output $key-$leaf]}
            }
            set f [open $log r];set text [read $f];close $f
            set fault 66564
            set marker "AXI4_FORWARD_SUITE_PASS kind=$kind device_period_ps=$period fault=$fault native_endpoint=0"
            if {[string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "AXI4 request case failed $key"}
            puts "AXI4_FORWARD_CASE_EXPORTED key=$key"
        }
    }
    close_project
    puts NTRPU_AXI4_FORWARD_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_section_bank [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
