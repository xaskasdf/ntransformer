set root [file normalize [file join [file dirname [info script]] ..]]
proc reserved_test {root output simulate_only} {
    if {[file exists $output]} {error "Expected new output directory"}
    file mkdir $output
    set_param general.maxThreads 2
    exec python3 -B [file join $root tools prepare_ntrpu_reserved_receiver.py] [file join $output generated]
    create_project reserved_dispatch [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic
                  ntrpu_destination_dispatch ntrpu_reserved_dispatch} {
        add_files [file join $root fpga rtl ${name}.sv]
    }
    add_files [file join $output generated ntrpu_peer_multibuffer_offer.sv]
    add_files [file join $output generated ntrpu_peer_multibuffer_reserved.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_reserved_dispatch.sv]
    set_property top tb_ntrpu_reserved_dispatch [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output project reserved_dispatch.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "RS SUITE_PASS cases=8 " $text]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Reserved dispatch simulation failed"}
    if {$simulate_only} {
        puts NTRPU_RESERVED_DISPATCH_SIMULATED
        close_project
        return
    }
    set_property top ntrpu_reserved_dispatch [get_filesets sources_1]
    read_xdc [file join $root fpga constraints ntrpu_multibuffer_ooc.xdc]
    synth_design -top ntrpu_reserved_dispatch -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
    report_utilization -file [file join $output utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing_synth.rpt]
    write_checkpoint [file join $output reserved_dispatch_synth.dcp]
    set f [open [file join $output structure.txt] w]
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    close $f
    puts NTRPU_RESERVED_DISPATCH_EXPORTED
    close_project
}
if {[llength $argv]<1 || [llength $argv]>2} {error "Expected new output directory and optional simulation-only flag"}
set simulate_only 0
if {[llength $argv]==2} {
    if {[lindex $argv 1] ne "simulation-only"} {error "Unknown mode"}
    set simulate_only 1
}
if {[catch {reserved_test $root [file normalize [lindex $argv 0]] $simulate_only} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
