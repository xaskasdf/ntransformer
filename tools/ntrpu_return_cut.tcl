# Isolated integration. Reuse the reference reports and bank-region procedures.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set text [read $f];close $f;return $text}
proc run_axi {root output stage} {
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    if {$stage in {rtl synth netlist}} {
        set cmd [list python3 -B [file join $root tools prepare_ntrpu_return_cut.py] $generated]
        if {$stage eq "netlist"} {lappend cmd --netlist [file join $output .. inputs physical_funcsim.v]}
        exec {*}$cmd
        create_project transport_axi [file join $output project] -part xcvu5p-flva2104-2-i
        source [file join $root tools ntrpu_overlap_axi_ip.tcl]
        if {$stage eq "netlist"} {set ipname ref_ntrpu_mem_regslice} else {set ipname ntrpu_mem_regslice}
        set ip [create_transport_axi_slice $output $ipname]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        file copy [get_property IP_FILE $ip] [file join $output ip.xci]
        if {$stage eq "netlist"} {
            foreach f [glob [file join $generated ref_*.sv]] {add_files $f}
            add_files [file join $output .. inputs physical_funcsim.v]
        } else {
            foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic} {
                add_files [file join $root fpga rtl ${n}.sv]
            }
            foreach n {ntrpu_buffer_pool_allocator_return_cut ntrpu_candidate_transport ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch ntrpu_transport_physical} {
                add_files [file join $generated ${n}.sv]
            }
        }
    }
    if {$stage in {rtl netlist}} {
        if {$stage eq "rtl"} {
            set tests {tb_ntrpu_candidate_transport protocol.log {TE SUITE_PASS cases=13 } tb_ntrpu_transport_overlap_extra extra.log {OV EXTRA_PASS cases=3} tb_ntrpu_transport_perf perf.log {PF ALL_PASS cases=15 } tb_ntrpu_return_cut_pool pool.log {PASS seed=} tb_ntrpu_return_cut_bounds bounds.log {PASS seed=} tb_ntrpu_return_cut_wrap wrap.log {PASS seed=}}
        } else {
            set tests {tb_ntrpu_transport_netlist protocol.log {TN ALL_PASS checks=} tb_ntrpu_overlap_netlist_extra extra.log {TN ALL_PASS checks=}}
        }
        foreach {top filename marker} $tests {
            add_files -fileset sim_1 [file join $generated ${top}.sv]
            set_property top $top [get_filesets sim_1]
            set_property xsim.elaborate.debug_level off [get_filesets sim_1]
            set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project transport_axi.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output $filename]
            set text [read_text $log]
            if {[string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Integrated simulation failed: $filename"}
        }
        close_project
    } else {
        if {$stage eq "synth"} {
            synth_design -top ntrpu_transport_physical -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
            create_clock -name transport_clk -period 5.000 [get_ports clk]
            set_clock_uncertainty -setup 0.100 [get_clocks transport_clk]
            set_clock_uncertainty -hold 0.050 [get_clocks transport_clk]
            set_property LOC BUFGCE_X0Y60 [get_cells parent_clock]
        } else {
            if {$stage eq "place"} {set prior synth} else {set prior place}
            open_checkpoint [file join $output .. inputs physical_${prior}.dcp]
            if {[get_property PERIOD [get_clocks transport_clk]]!=5.0 || [get_property LOC [get_cells parent_clock]] ne "BUFGCE_X0Y60"} {error "Clock changed"}
            write_xdc -constraints all [file join $output constraints_before.xdc]
            if {$stage eq "place"} {opt_design;place_design} else {
                set failed [catch {route_design} message options]
                set f [open [file join $output route_command.txt] w];puts $f "RETURN_CODE $failed\nMESSAGE $message";close $f
            }
        }
        set reference [read_text [file join $root tools ntrpu_transport_study.tcl]]
        set begin [string first {        write_checkpoint [file join $output physical_${stage}.dcp]} $reference]
        set end [string first {        close_design} $reference $begin]
        if {$begin<0 || $end<$begin} {error "Physical report boundary changed"}
        set reports [string range $reference $begin [expr {$end-1}]]
        set needle {        write_checkpoint [file join $output physical_${stage}.dcp]}
        eval $reports
        close_design
    }
    puts "NTRPU_RETURN_CUT_COMPLETE $stage"
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {rtl synth place route netlist}} {error "Expected output and stage"}
if {[catch {run_axi $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
