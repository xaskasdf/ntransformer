# One cold epoch, native HBM plus unchanged two-clock transport and adapters.
set_param general.maxThreads 2
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
set reference [file normalize [lindex $argv 1]]
set mode [lindex $argv 2]
proc read_text {p} {set f [open $p r];set t [read $f];close $f;return $t}
if {[catch {
    file mkdir $out
    create_project hbm_adapter [file join $out project] -part xcvu47p-fsvh2892-2-e
    set original [read_text [file join $root tools ntrpu_hbm_adapter.tcl]]
    set begin [string first {proc save_ip} $original]
    set end [string first {proc hbm_adapter} $original]
    eval [string range $original $begin [expr {$end-1}]]
    set begin [string first {    create_ip -name axi_dwidth_converter} $original]
    set end [string first {    if {$stage eq "sim"}} $original $begin]
    if {$begin<0 || $end<$begin} {error "Qualified IP configuration boundary changed"}
    set stage ip
    set output $out
    eval [string range $original $begin [expr {$end-1}]]
    source [file join $root tools ntrpu_overlap_axi_ip.tcl]
    set ip [create_transport_axi_slice $out]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name hbm_cdc
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {WRITE_ONLY} \
        CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {4} \
        CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3}] [get_ips hbm_cdc]
    generate_target all [get_ips hbm_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips hbm_cdc]]]
    save_ip $out hbm_cdc
    if {$mode eq "compile"} {
        create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name native_hbm
        set_property -dict [list CONFIG.USER_HBM_DENSITY {16GB} CONFIG.USER_SWITCH_ENABLE_00 {TRUE} \
            CONFIG.USER_SWITCH_ENABLE_01 {TRUE} CONFIG.USER_AXI_CLK_FREQ {450} CONFIG.USER_AXI_CLK1_FREQ {450}] [get_ips native_hbm]
        save_ip $out native_hbm
        generate_target all [get_ips native_hbm]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips native_hbm]]]
        set ipdir [file join $out project hbm_adapter.gen sources_1 ip native_hbm]
        file copy [file join $ipdir native_hbm.veo] [file join $out native_hbm.veo]
        file copy [get_property IP_FILE [get_ips native_hbm]] [file join $out native_hbm.xci]
        file mkdir [file join $out model_inputs]
        foreach p [glob -nocomplain [file join $ipdir hdl rtl *.mem]] {file copy $p [file join $out model_inputs [file tail $p]]}
    } elseif {$mode eq "preflight"} {
        file copy [file join $root fixture native_hbm.veo] [file join $out native_hbm.veo]
    } else {error "Unknown mode"}
    exec python3 -B [file join $root tools prepare_ntrpu_hbm_native_transport.py] $out $reference
    foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch ntrpu_buffer_pool_allocator_return_cut} {
        add_files [file join $out generated ${n}.sv]
    }
    foreach n {ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter ntrpu_hbm_startup ntrpu_hbm_transport_async} {
        add_files [file join $root fpga rtl ${n}.sv]
    }
    if {$mode eq "compile"} {add_files [file join $out ntrpu_hbm_native_rw.sv]} else {add_files [file join $out generated ntrpu_hbm_native_stub.sv]}
    add_files -fileset sim_1 [file join $out generated tb_ntrpu_hbm_native_transport.sv]
    set_property top tb_ntrpu_hbm_native_transport [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation -step compile
    launch_simulation -step elaborate
    if {$mode eq "preflight"} {
        launch_simulation -step simulate
        run all
        close_sim
        file copy [file join $out project hbm_adapter.sim sim_1 behav xsim simulate.log] [file join $out simulation.log]
        set log [read_text [file join $out simulation.log]]
        if {[string first {NC ALL_PASS cases=4 } $log]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $log]} {error "Native transport preflight failed"}
    }
    close_project
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
puts "NTRPU_NATIVE_TRANSPORT_COMPLETE $mode"
exit 0
