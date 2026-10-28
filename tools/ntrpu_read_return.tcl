set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc prepare_read_return {root output} {
    set_param general.maxThreads 2
    create_project hbm_adapter [file join $output project] -part xcvu47p-fsvh2892-2-e
    set original [read_text [file join $root tools ntrpu_hbm_adapter.tcl]]
    set begin [string first {proc save_ip} $original];set end [string first {proc hbm_adapter} $original]
    eval [string range $original $begin [expr {$end-1}]]
    set begin [string first {    create_ip -name axi_dwidth_converter} $original]
    set end [string first {    if {$stage eq "sim"}} $original $begin]
    if {$begin<0 || $end<$begin} {error "Qualified write configuration boundary changed"}
    set stage ip;eval [string range $original $begin [expr {$end-1}]]
    create_ip -vlnv xilinx.com:ip:axi_dwidth_converter:2.1 -module_name hbm_read_width
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {READ_ONLY} \
        CONFIG.ADDR_WIDTH {34} CONFIG.SI_DATA_WIDTH {512} CONFIG.MI_DATA_WIDTH {256} \
        CONFIG.SI_ID_WIDTH {4} CONFIG.MAX_SPLIT_BEATS {16} CONFIG.ACLK_ASYNC {0} CONFIG.FIFO_MODE {0}] [get_ips hbm_read_width]
    create_ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 -module_name hbm_read_protocol
    set_property -dict [list CONFIG.SI_PROTOCOL {AXI4} CONFIG.MI_PROTOCOL {AXI3} \
        CONFIG.READ_WRITE_MODE {READ_ONLY} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} \
        CONFIG.ID_WIDTH {4} CONFIG.TRANSLATION_MODE {2}] [get_ips hbm_read_protocol]
    foreach {name width mode} {hbm_cdc 64 WRITE_ONLY hbm_read_cdc 34 READ_ONLY} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE $mode CONFIG.DATA_WIDTH {512} \
            CONFIG.ADDR_WIDTH $width CONFIG.ID_WIDTH {4} CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3}] [get_ips $name]
    }
    foreach name {hbm_cdc hbm_read_cdc hbm_read_width hbm_read_protocol} {
        generate_target all [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips $name]]]
        save_ip $output $name
        file copy [file join $output project hbm_adapter.gen sources_1 ip $name sim ${name}.v] [file join $output ${name}_wrapper.v]
    }
    foreach n {ntrpu_rv_elastic ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter ntrpu_hbm_startup} {add_files [file join $root fpga rtl ${n}.sv]}
    foreach n {ntrpu_locality_allocator ntrpu_dual_ingress_locality ntrpu_dual_hbm_async ntrpu_hbm_read_frame ntrpu_hbm_read_adapter ntrpu_hbm_read_return} {add_files [file join $root fpga experimental ${n}.sv]}
}
proc simulate_read_return {root output common native} {
    prepare_read_return $root $output
    add_files -fileset sim_1 [file join $output native_stub.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_read_return.sv]
    set_property include_dirs [list $output] [get_filesets sim_1]
    set_property top tb_ntrpu_read_return [get_filesets sim_1]
    set_property generic "COMMON=$common NATIVE=$native" [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation;run all;close_sim
    set log [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set text [read_text $log];set cases [expr {$native?4:20}]
    if {[string first "RR ALL_PASS cases=$cases " $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Read return test failed"}
    close_project
    puts NTRPU_READ_RETURN_COMPLETE
}
if {[llength $argv]!=3} {error "Expected output, common-clock flag, native-profile flag"}
if {[catch {simulate_read_return $root [file normalize [lindex $argv 0]] [lindex $argv 1] [lindex $argv 2]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
