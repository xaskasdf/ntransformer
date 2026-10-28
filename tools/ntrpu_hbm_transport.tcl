# Isolated transport composition; same-clock functional or OOC measurement.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc run_composition {root output action} {
    set_param general.maxThreads 2
    create_project hbm_adapter [file join $output project] -part xcvu5p-flva2104-2-i
    set f [open [file join $output hbm_inventory.txt] w]
    puts $f "CURRENT_PART [get_property PART [current_project]]"
    puts $f "HBM_IP [get_ipdefs -all xilinx.com:ip:hbm:*]"
    foreach pattern {xcvu31p* xcvu33p* xcvu35p* xcvu37p* xcvu45p* xcvu47p* xcvu57p*} {
        puts $f "$pattern [lsort [get_parts -quiet $pattern]]"
    }
    close $f
    if {$action ne "inventory"} {
        # Reuse exactly the qualified IP configuration without changing its file.
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
        foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch ntrpu_buffer_pool_allocator_return_cut} {
            add_files [file join $output generated ${n}.sv]
        }
        foreach n {ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter ntrpu_hbm_transport} {
            add_files [file join $root fpga rtl ${n}.sv]
        }
        if {$action eq "sim"} {
            add_files -fileset sim_1 [file join $output generated tb_ntrpu_hbm_transport.sv]
            set_property top tb_ntrpu_hbm_transport [get_filesets sim_1]
            set_property xsim.elaborate.debug_level off [get_filesets sim_1]
            set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
            launch_simulation;run all;close_sim
            set log [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output simulation.log]
            set text [read_text $log]
            if {[string first {HT ALL_PASS cases=17 } $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "HBM transport simulation failed"}
        } else {
            synth_design -top ntrpu_hbm_transport -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
            create_clock -name transport_clk -period 5.000 [get_ports clk]
            set_clock_uncertainty -setup 0.100 [get_clocks transport_clk]
            set_clock_uncertainty -hold 0.050 [get_clocks transport_clk]
            write_checkpoint [file join $output composition_synth.dcp]
            report_utilization -hierarchical -file [file join $output utilization.rpt]
            report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
            report_timing -from [all_registers] -to [all_registers] -delay_type max -max_paths 10 -file [file join $output internal_setup.rpt]
            report_timing -from [all_registers] -to [all_registers] -delay_type min -max_paths 10 -file [file join $output internal_hold.rpt]
            report_drc -file [file join $output drc.rpt]
            check_timing -verbose -file [file join $output check_timing.rpt]
            report_exceptions -coverage -file [file join $output exceptions.rpt]
            write_xdc -constraints all [file join $output constraints.xdc]
            set f [open [file join $output metrics.txt] w]
            puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
            puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
            puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
            puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
            puts $f "CLOCKS [llength [get_clocks]]"
            puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks transport_clk]]"
            close $f
        }
    }
    close_project
    puts "NTRPU_HBM_TRANSPORT_COMPLETE $action"
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {sim synth inventory}} {error "Expected output and stage"}
if {[catch {run_composition $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
