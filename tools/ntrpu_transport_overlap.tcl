# Isolated behavioral experiments; no changes to the reference RTL or queues.
set root [file normalize [file join [file dirname [info script]] ..]]
proc overlap {root output variant} {
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    exec python3 -B [file join $root tools prepare_ntrpu_transport_overlap.py] $generated --variant $variant
    create_project overlap [file join $output project] -part xcvu5p-flva2104-2-i
    foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport} {
        add_files [file join $root fpga rtl ${n}.sv]
    }
    foreach n {ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch} {
        add_files [file join $generated ${n}.sv]
    }
    foreach n {tb_ntrpu_candidate_transport tb_ntrpu_transport_perf tb_ntrpu_transport_overlap_extra} {
        add_files -fileset sim_1 [file join $generated ${n}.sv]
    }
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_transport_overlap_credit.sv]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach {label top marker} {
        credit tb_ntrpu_transport_overlap_credit {OC SUITE_PASS cases=5}
        protocol tb_ntrpu_candidate_transport {TE SUITE_PASS cases=13 }
        extra tb_ntrpu_transport_overlap_extra {OV EXTRA_PASS cases=3}
        perf tb_ntrpu_transport_perf {PF ALL_PASS cases=15 }
    } {
        set_property top $top [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project overlap.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${label}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Failed $label"}
    }
    close_project
}
if {[catch {overlap $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
puts "NTRPU_OVERLAP_COMPLETE [lindex $argv 1]"
exit 0
