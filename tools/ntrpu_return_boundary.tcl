# Boundary sensitivity only. The registered-return circuit is unchanged.
set root [file normalize [file join [file dirname [info script]] ..]]
proc run_boundary {root output} {
    set_param general.maxThreads 2
    file mkdir $output
    set generated [file join $output generated]
    exec python3 -B [file join $root tools prepare_ntrpu_return_boundary.py] $generated
    create_project return_boundary [file join $output project] -part xcvu5p-flva2104-2-i
    source [file join $root tools ntrpu_overlap_axi_ip.tcl]
    set ip [create_transport_axi_slice $output]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic} {
        add_files [file join $root fpga rtl ${n}.sv]
    }
    foreach n {ntrpu_buffer_pool_allocator_return_cut ntrpu_candidate_transport ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch} {
        add_files [file join $generated ${n}.sv]
    }
    add_files -fileset sim_1 [file join $generated tb_ntrpu_return_boundary.sv]
    set_property top tb_ntrpu_return_boundary [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation;run all;close_sim
    set log [file join $output project return_boundary.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output boundary.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first {BN ALL_PASS cases=45 } $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Boundary simulation failed"}
    close_project
    puts "NTRPU_RETURN_BOUNDARY_COMPLETE"
}
if {[llength $argv]!=1} {error "Expected output"}
if {[catch {run_boundary $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
