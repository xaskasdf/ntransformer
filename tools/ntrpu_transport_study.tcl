# Isolated performance / physical stages. No exceptions or sampler registers.
set root [file normalize [file join [file dirname [info script]] ..]]
proc run_study {root output stage} {
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    if {$stage in {perf synth netlist}} {
        set cmd [list python3 -B [file join $root tools prepare_ntrpu_transport_study.py] $generated]
        if {$stage eq "netlist"} {lappend cmd --netlist [file join $output .. inputs physical_funcsim.v]}
        exec {*}$cmd
    }
    if {$stage in {perf netlist}} {
        create_project transport_study [file join $output project] -part xcvu5p-flva2104-2-i
        if {$stage eq "perf"} {
            foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport} {
                add_files [file join $root fpga rtl ${n}.sv]
            }
            foreach n {ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch} {
                add_files [file join $generated ${n}.sv]
            }
            set top tb_ntrpu_transport_perf
            set marker {PF ALL_PASS cases=15 }
        } else {
            foreach f [glob [file join $generated ref_*.sv]] {add_files $f}
            add_files [file join $output .. inputs physical_funcsim.v]
            set top tb_ntrpu_transport_netlist
            set marker {TN ALL_PASS checks=}
        }
        add_files -fileset sim_1 [file join $generated ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set_property xsim.elaborate.debug_level off [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project transport_study.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output simulation.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Study simulation failed"}
        close_project
    } else {
        if {$stage eq "synth"} {
            create_project -in_memory -part xcvu5p-flva2104-2-i
            foreach n {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_rv_elastic ntrpu_candidate_transport} {
                read_verilog -sv [file join $root fpga rtl ${n}.sv]
            }
            foreach n {ntrpu_destination_candidate ntrpu_peer_multibuffer_offer ntrpu_peer_multibuffer_reserved ntrpu_candidate_dispatch ntrpu_transport_physical} {
                read_verilog -sv [file join $generated ${n}.sv]
            }
            synth_design -top ntrpu_transport_physical -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
            create_clock -name transport_clk -period 5.000 [get_ports clk]
            set_clock_uncertainty -setup 0.100 [get_clocks transport_clk]
            set_clock_uncertainty -hold 0.050 [get_clocks transport_clk]
            set_property LOC BUFGCE_X0Y60 [get_cells parent_clock]
        } else {
            if {$stage eq "place"} {set prior synth} else {set prior place}
            open_checkpoint [file join $output .. inputs physical_${prior}.dcp]
            if {[get_property PERIOD [get_clocks transport_clk]]!=5.0 || [get_property LOC [get_cells parent_clock]] ne "BUFGCE_X0Y60"} {error "Clock context changed"}
            write_xdc -constraints all [file join $output constraints_before.xdc]
            if {$stage eq "place"} {opt_design;place_design} else {
                set failed [catch {route_design} message options]
                set f [open [file join $output route_command.txt] w];puts $f "RETURN_CODE $failed\nMESSAGE $message";close $f
            }
        }
        write_checkpoint [file join $output physical_${stage}.dcp]
        report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type max -max_paths 10 -file [file join $output internal_setup.rpt]
        report_timing -from [all_registers] -to [all_registers] -delay_type min -max_paths 10 -file [file join $output internal_hold.rpt]
        report_clocks -file [file join $output clocks.rpt]
        report_utilization -file [file join $output utilization.rpt]
        report_drc -file [file join $output drc.rpt]
        report_exceptions -coverage -file [file join $output exceptions.rpt]
        write_xdc -constraints all [file join $output constraints_after.xdc]
        set f [open [file join $output drc_violations.tsv] w];puts $f "name\tseverity"
        foreach v [get_drc_violations -quiet] {puts $f "[get_property NAME $v]\t[get_property SEVERITY $v]"};close $f
        set f [open [file join $output metrics.txt] w]
        puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks transport_clk]]"
        puts $f "CLOCKS [llength [get_clocks]]"
        puts $f "CLOCK_LOCATION [get_property LOC [get_cells parent_clock]]"
        puts $f "BUFGCE [llength [get_cells -hier -filter {REF_NAME == BUFGCE}]]"
        puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
        puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
        puts $f "INTERNAL_SETUP [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type max -max_paths 1]]"
        puts $f "INTERNAL_HOLD [get_property SLACK [get_timing_paths -from [all_registers] -to [all_registers] -delay_type min -max_paths 1]]"
        close $f
        if {$stage eq "route"} {
            report_route_status -file [file join $output route_status.rpt]
            check_timing -verbose -file [file join $output check_timing.rpt]
            report_methodology -file [file join $output methodology.rpt]
            # Expand any boundary routing warning: no hidden internal load may
            # be dismissed as an unqualified external connection.
            set f [open [file join $output boundary_nets.tsv] w]
            puts $f "net\troute_status\tdrivers\tloads\tports"
            foreach net [lsort -unique [get_nets -quiet -of_objects [get_drc_violations -quiet *RTSTAT-10*]]] {
                set seg [get_nets -segments $net]
                set drivers [get_pins -quiet -leaf -of_objects $seg -filter {DIRECTION == OUT}]
                set loads [get_pins -quiet -leaf -of_objects $seg -filter {DIRECTION == IN}]
                set ports [get_ports -quiet -of_objects $seg]
                puts $f "$net\t[get_property ROUTE_STATUS $net]\t[join [lsort $drivers] {;}]\t[join [lsort $loads] {;}]\t[join [lsort $ports] {;}]"
            }
            close $f
            write_verilog -mode funcsim [file join $output physical_funcsim.v]
            if {$failed} {error "Router failed; diagnostic checkpoint preserved"}
        }
        close_design
    }
    puts "NTRPU_TRANSPORT_STUDY_COMPLETE $stage"
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {perf synth place route netlist}} {error "Expected output and stage"}
if {[catch {run_study $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
