# Functional CDC integration; no hard HBM instance in this stage.
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc run_async {root output} {
    set_param general.maxThreads 2
    create_project hbm_adapter [file join $output project] -part xcvu47p-fsvh2892-2-e
    # Keep the qualified width/protocol and transport-slice IP settings intact.
    set original [read_text [file join $root tools ntrpu_hbm_adapter.tcl]]
    set begin [string first {proc save_ip} $original]
    set end [string first {proc hbm_adapter} $original]
    eval [string range $original $begin [expr {$end-1}]]
    set begin [string first {    create_ip -name axi_dwidth_converter} $original]
    set end [string first {    if {$stage eq "sim"}} $original $begin]
    if {$begin<0 || $end<$begin} {error "IP configuration boundary changed"}
    set stage ip
    eval [string range $original $begin [expr {$end-1}]]
    source [file join $root tools ntrpu_overlap_axi_ip.tcl]
    set ip [create_transport_axi_slice $output]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name hbm_cdc
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {WRITE_ONLY} \
        CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {4} \
        CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3}] [get_ips hbm_cdc]
    generate_target all [get_ips hbm_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips hbm_cdc]]]
    save_ip $output hbm_cdc
    foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch ntrpu_buffer_pool_allocator_return_cut} {
        add_files [file join $output generated ${n}.sv]
    }
    foreach n {ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter ntrpu_hbm_startup ntrpu_hbm_transport_async} {
        add_files [file join $root fpga rtl ${n}.sv]
    }
    add_files -fileset sim_1 [file join $output generated tb_ntrpu_hbm_cdc.sv]
    set_property top tb_ntrpu_hbm_cdc [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation;run all;close_sim
    set log [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set text [read_text $log]
    if {[string first {CD ALL_PASS cases=21 } $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "HBM asynchronous integration failed"}
    close_project
    puts "NTRPU_HBM_CDC_COMPLETE async"
}
