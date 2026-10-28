set root [file normalize [file join [file dirname [info script]] ..]]
proc drain_sections {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project sections $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_drain_certificate_join ntrpu_local_reply_certificate ntrpu_tagged_reset_mailbox ntrpu_axi4_section_certificate_r2 ntrpu_axi4_retained_admission_r2 ntrpu_axi4_response_guard_r2 ntrpu_axi4_pair_drain_r2 ntrpu_axi4_drain_monitor_r2 ntrpu_axil_cdc_recode ntrpu_axil_pair_drain ntrpu_axil_drain_monitor ntrpu_reset_supervisor ntrpu_local_reset_actuator ntrpu_axil_retained_admission_r2 ntrpu_axil_response_guard ntrpu_axil_fence_certificate_r5 ntrpu_axil_fence_section_r3} {
        add_files [file join $root fpga rtl $module.sv]
    }
    foreach {name protocol width address id} {ntrpu_axi4_cdc_probe AXI4 512 64 4 ntrpu_host_axil_cdc AXI4LITE 32 32 0} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set properties [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend properties CONFIG.ID_WIDTH $id}
        set_property -dict $properties $ip
        generate_target all $ip
        report_property -all -file [file join $output $name.rpt] $ip
    }
    create_ip -vlnv xilinx.com:ip:dfx_axi_shutdown_manager:1.0 -module_name ntrpu_host_axil_shutdown
    set ip [get_ips ntrpu_host_axil_shutdown]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.RESET_ACTIVE_LEVEL 0 CONFIG.RP_IS_MASTER false CONFIG.CTRL_INTERFACE_TYPE 0 CONFIG.DP_PROTOCOL AXI4LITE CONFIG.DP_AXI_ADDR_WIDTH 32 CONFIG.DP_AXI_DATA_WIDTH 32 CONFIG.DP_AXI_RESP 0} $ip
    generate_target all $ip
    report_property -all -file [file join $output shutdown_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    foreach module {ntrpu_drain_axil_fixture ntrpu_drain_axi4_fixture tb_ntrpu_drain_sections} {
        add_files -fileset sim_1 [file join $root fpga tb $module.sv]
    }
    set_property top tb_ntrpu_drain_sections [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach period {5000 37000} {
        foreach external {0 1} {
            set key sections-$external-$period
            set_property generic [list DEVICE_PERIOD_PS=$period EXTERNAL_FIXTURE=$external] [get_filesets sim_1]
            # Compile through generated native scripts, then use a fresh XSIM
            # process per profile. Never reuse the IDE's VCD object registry.
            launch_simulation -scripts_only
            set simdir [file join $output sections.sim sim_1 behav xsim]
            set capture [file join $simdir tb_ntrpu_drain_sections.tcl]
            set f [open $capture w]
            puts $f [list open_vcd [file join $output $key.vcd]]
            foreach scope {/tb_ntrpu_drain_sections/* /tb_ntrpu_drain_sections/lite/* /tb_ntrpu_drain_sections/axi4/* /tb_ntrpu_drain_sections/local_lite/* /tb_ntrpu_drain_sections/local_axi4/*} {
                puts $f "set objects \[get_objects $scope\]"
                puts $f {if {[llength $objects]==0} {error "Missing VCD objects"}}
                puts $f {log_vcd $objects}
            }
            puts $f {run all}
            puts $f {flush_vcd}
            puts $f {close_vcd}
            puts $f {quit}
            close $f
            set previous [pwd]
            cd $simdir
            exec bash compile.sh >[file join $output $key-compile-console.log] 2>@1
            exec bash elaborate.sh >[file join $output $key-elaborate-console.log] 2>@1
            exec bash simulate.sh >[file join $output $key-console.log] 2>@1
            cd $previous
            set f [open [file join $output $key.vcd] r];set wave [read $f];close $f
            if {[regexp -all {\$var } $wave]<100 || [file size [file join $output $key.vcd]]<10000} {error "Empty/incomplete VCD $key"}
            foreach leaf {simulate.log elaborate.log xvlog.log} {file copy [file join $simdir $leaf] [file join $output $key-$leaf]}
            set f [open [file join $simdir simulate.log] r];set text [read $f];close $f
            set expected "DRAIN_SECTIONS_PASS external=$external period=$period actual_sections=2 local_providers=2 native_endpoint=0 external_providers_qualified=0 reset_driven=0"
            if {[lsearch -exact [split $text \n] $expected]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $text]} {error "Section integration failed $key"}
            puts "DRAIN_SECTIONS_PROFILE_EXPORTED key=$key"
        }
    }
    close_project
    puts NTRPU_DRAIN_SECTIONS_R2_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {drain_sections $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
