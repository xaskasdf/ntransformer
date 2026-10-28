# Isolated functional validation and paired synthesis. No automatic long routing.
set root [file normalize [file join [file dirname [info script]] ..]]
proc run_candidate {root output} {
    if {[file exists $output]} {error "Expected a new output directory"}
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    exec python3 -B [file join $root tools prepare_ntrpu_candidate_dispatch.py] $generated
    create_project candidate_dispatch [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_registered_dispatch} {
        add_files [file join $root fpga rtl ${name}.sv]
    }
    foreach name {ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_destination_registered
                  ntrpu_destination_candidate ntrpu_candidate_dispatch ntrpu_registered_fixture ntrpu_candidate_fixture} {
        add_files [file join $generated ${name}.sv]
    }
    foreach {top marker name} {
        tb_ntrpu_candidate_dispatch {RS SUITE_PASS cases=8 } simulation
        tb_ntrpu_candidate_extra {CP EXTRA_PASS cases=2 } extra
    } {
        add_files -fileset sim_1 [file join $generated ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project candidate_dispatch.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {
            error "Candidate $name simulation failed"
        }
    }
    close_project
    foreach variant {registered candidate} {
        create_project -in_memory -part xcvu5p-flva2104-2-i
        foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_registered_dispatch} {
            read_verilog -sv [file join $root fpga rtl ${name}.sv]
        }
        foreach name {ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_destination_registered
                      ntrpu_destination_candidate ntrpu_candidate_dispatch} {
            read_verilog -sv [file join $generated ${name}.sv]
        }
        read_verilog -sv [file join $generated ntrpu_${variant}_fixture.sv]
        set top ntrpu_${variant}_fixture
        # Timing sampler ports are deliberately unqualified. All core ports
        # terminate at same-clock registers; only register-to-register paths
        # are evidence about the core. No false/multicycle-path exceptions.
        set xdc [file join $output ${variant}_context.xdc]
        set f [open $xdc w]
        puts $f {create_clock -name sample_clk -period 5.000 [get_ports clk]}
        puts $f {set_clock_uncertainty -setup 0.100 [get_clocks sample_clk]}
        puts $f {set_clock_uncertainty -hold 0.050 [get_clocks sample_clk]}
        close $f
        read_xdc $xdc
        synth_design -top $top -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
        report_utilization -file [file join $output ${variant}_utilization.rpt]
        report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output ${variant}_timing_synth.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type max -max_paths 10 -file [file join $output ${variant}_internal_setup.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type min -max_paths 10 -file [file join $output ${variant}_internal_hold.rpt]
        set f [open [file join $output ${variant}_structure.txt] w]
        puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks sample_clk]]"
        puts $f "BUFGCE [llength [get_cells -hier -filter {REF_NAME == BUFGCE}]]"
        puts $f "BOUNDARY_REGISTERS [llength [get_cells -hier -filter {NAME =~ boundary_* && REF_NAME =~ FD*}]]"
        puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
        puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
        puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
        puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
        close $f
        write_checkpoint [file join $output ${variant}_synth.dcp]
        close_project
    }
    puts NTRPU_CANDIDATE_PAIRED_SYNTH_COMPLETE
}
if {[llength $argv]!=1} {error "Expected output directory"}
if {[catch {run_candidate $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
