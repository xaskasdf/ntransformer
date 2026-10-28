set_param general.maxThreads 2
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
set stage [lindex $argv 1]
set mode [lindex $argv 2]
if {[catch {
    create_project dual_ingress [file join $out project] -part xcvu47p-fsvh2892-2-e
    add_files [file join $root fpga experimental ntrpu_dual_allocator.sv]
    add_files [file join $root fpga experimental ntrpu_dual_ingress.sv]
    if {$stage eq "sim"} {
        add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_dual_ingress.sv]
        set_property top tb_ntrpu_dual_ingress [get_filesets sim_1]
        set_property generic SERIAL_INPUT=$mode [get_filesets sim_1]
        set_property xsim.elaborate.debug_level off [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        file copy [file join $out project dual_ingress.sim sim_1 behav xsim simulate.log] [file join $out simulation.log]
        set f [open [file join $out simulation.log]];set log [read $f];close $f
        if {[string first "DI ALL_PASS mode=$mode cases=8 " $log]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:)} $log]} {error "Dual ingress simulation failed"}
    } elseif {$stage eq "synth"} {
        synth_design -top ntrpu_dual_ingress -mode out_of_context -generic SERIAL_INPUT=$mode -part xcvu47p-fsvh2892-2-e
        create_clock -name source -period 5 [get_ports clk]
        report_utilization -file [file join $out utilization.rpt]
        report_utilization -hierarchical -file [file join $out hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $out timing.rpt]
        check_timing -verbose -file [file join $out check_timing.rpt]
        report_drc -file [file join $out drc.rpt]
        write_checkpoint [file join $out dual_ingress.dcp]
        set f [open [file join $out structure.tsv] w]
        foreach pattern {RAMB36* RAMB18* URAM* LD*} {
            set filter [format {REF_NAME =~ %s} $pattern]
            puts $f "$pattern\t[llength [get_cells -hierarchical -filter $filter]]"
        }
        puts $f "blackboxes\t[llength [get_cells -hierarchical -filter {IS_BLACKBOX == 1}]]"
        close $f
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME =~ LD*}]]!=0} {error "Unexpected latches or blackboxes"}
    } else {error "Invalid stage"}
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts "DUAL_INGRESS_COMPLETE $stage $mode"
exit 0
