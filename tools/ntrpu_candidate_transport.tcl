# Functional transport boundary and synthesis sanity. No placement or routing.
set root [file normalize [file join [file dirname [info script]] ..]]
proc run_transport {root output simulation_only} {
    if {[file exists $output]} {error "Expected fresh output"}
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    exec python3 -B [file join $root tools prepare_ntrpu_candidate_dispatch.py] $generated
    create_project candidate_transport [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport} {
        add_files [file join $root fpga rtl ${name}.sv]
    }
    foreach name {ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_destination_candidate ntrpu_candidate_dispatch} {
        add_files [file join $generated ${name}.sv]
    }
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_candidate_transport.sv]
    set_property top tb_ntrpu_candidate_transport [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output project candidate_transport.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first {TE SUITE_PASS cases=13 } $text]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Transport simulation failed"}
    if {$simulation_only} {
        close_project
        puts NTRPU_CANDIDATE_TRANSPORT_SIM_COMPLETE
        return
    }
    set_property top ntrpu_candidate_transport [current_fileset]
    synth_design -top ntrpu_candidate_transport -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
    create_clock -name transport_clk -period 5.000 [get_ports clk]
    set_clock_uncertainty -setup 0.100 [get_clocks transport_clk]
    set_clock_uncertainty -hold 0.050 [get_clocks transport_clk]
    report_utilization -file [file join $output utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing_synth.rpt]
    report_drc -file [file join $output drc_synth.rpt]
    set f [open [file join $output structure.txt] w]
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    close $f
    write_checkpoint [file join $output transport_synth.dcp]
    close_project
    puts NTRPU_CANDIDATE_TRANSPORT_COMPLETE
}
if {[llength $argv]<1 || [llength $argv]>2} {error "Expected output and optional simulation-only"}
set simulation_only 0
if {[llength $argv]==2} {
    if {[lindex $argv 1] ne "simulation-only"} {error "Unknown mode"}
    set simulation_only 1
}
if {[catch {run_transport $root [file normalize [lindex $argv 0]] $simulation_only} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
