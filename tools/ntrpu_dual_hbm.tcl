set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc run_dual_hbm {root output policy native_profile} {
    set_param general.maxThreads 2
    create_project hbm_adapter [file join $output project] -part xcvu47p-fsvh2892-2-e
    set original [read_text [file join $root tools ntrpu_hbm_adapter.tcl]]
    set begin [string first {proc save_ip} $original]
    set end [string first {proc hbm_adapter} $original]
    eval [string range $original $begin [expr {$end-1}]]
    set begin [string first {    create_ip -name axi_dwidth_converter} $original]
    set end [string first {    if {$stage eq "sim"}} $original $begin]
    if {$begin<0 || $end<$begin} {error "Qualified IP configuration boundary changed"}
    set stage ip
    eval [string range $original $begin [expr {$end-1}]]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name hbm_cdc
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {WRITE_ONLY} \
        CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {4} \
        CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3}] [get_ips hbm_cdc]
    generate_target all [get_ips hbm_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips hbm_cdc]]]
    save_ip $output hbm_cdc
    foreach n {ntrpu_rv_elastic ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter ntrpu_hbm_startup} {
        add_files [file join $root fpga rtl ${n}.sv]
    }
    foreach n {ntrpu_locality_allocator ntrpu_dual_ingress_locality ntrpu_dual_hbm_async} {
        add_files [file join $root fpga experimental ${n}.sv]
    }
    add_files -fileset sim_1 [file join $output native_stub.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_dual_hbm.sv]
    set_property include_dirs [list $output] [get_filesets sim_1]
    set_property top tb_ntrpu_dual_hbm [get_filesets sim_1]
    set_property generic "PREFER_LOCAL=$policy NATIVE=$native_profile" [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation;run all;close_sim
    set log [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set text [read_text $log]
    set cases [expr {$native_profile?3:11}]
    if {[string first "DH ALL_PASS cases=$cases " $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Dual HBM integration failed"}
    close_project
    puts NTRPU_DUAL_HBM_COMPLETE
}
if {[llength $argv]!=3} {error "Expected output, locality policy, native profile"}
if {[catch {run_dual_hbm $root [file normalize [lindex $argv 0]] [lindex $argv 1] [lindex $argv 2]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
