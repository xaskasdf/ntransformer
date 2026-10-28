# OOC internal timing study: actual global buffers, no board IO timing claim.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set t [read $f];close $f;return $t}
proc physical {root output stage checkpoint} {
    set_param general.maxThreads 2
    if {$stage eq "synth"} {
        set text [read_text [file join $root tools ntrpu_hbm_cdc_async.tcl]]
        set begin [string first {    create_project hbm_adapter} $text]
        set end [string first {    add_files -fileset sim_1} $text]
        if {$begin<0 || $end<=$begin} {error "Frozen setup boundary changed"}
        eval [string range $text $begin [expr {$end-1}]]
        # The accepted IP setup uses a local variable named stage, set to ip.
        set stage synth
        add_files [file join $output generated ntrpu_hbm_physical.sv]
        set xdc [file join $output clocks.xdc]
        set f [open $xdc w]
        puts $f {create_clock -name transport_clk -period 5.000 [get_ports clk]}
        puts $f {create_clock -name memory_clk -period 2.222 [get_ports mem_clk]}
        puts $f {create_clock -name apb_status_clk -period 10.000}
        puts $f {set_input_delay -clock apb_status_clk -min 0.000 [get_ports init_complete_async*]}
        puts $f {set_input_delay -clock apb_status_clk -max 0.000 [get_ports init_complete_async*]}
        puts $f {set_clock_uncertainty -setup 0.100 [get_clocks {transport_clk memory_clk}]}
        puts $f {set_clock_uncertainty -hold 0.050 [get_clocks {transport_clk memory_clk}]}
        close $f
        add_files -fileset constrs_1 $xdc
        synth_design -top ntrpu_hbm_physical -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
        read_xdc [file join $root fpga constraints ntrpu_hbm_startup_cdc.xdc]
        if {[llength $ntrpu_status_internal_d]!=3 || [llength $ntrpu_status_apb_d]!=2} {error "Status constraints missed endpoints"}
    } else {
        open_checkpoint $checkpoint
        if {$stage eq "place"} {opt_design;place_design}
        if {$stage eq "route"} {route_design}
    }
    # Save expensive implementation work before any reporting/inspection gate.
    write_checkpoint [file join $output physical_${stage}.dcp]
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Part mismatch"}
    foreach c {source_clock memory_clock} {
        if {[llength [get_cells -quiet $c]]!=1 || [get_property REF_NAME [get_cells $c]] ne "BUFGCE"} {error "Missing explicit clock buffer $c"}
    }
    # Vivado may insert additional global buffers on high-fanout control nets.
    set buffers [get_cells -hier -filter {REF_NAME =~ BUFG*}]
    set f [open [file join $output buffers.tsv] w]
    foreach c [lsort $buffers] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property LOC $c]\t[get_nets -of_objects [get_pins $c/O]]\t[get_clocks -of_objects [get_pins $c/O]]"
    };close $f
    set f [open [file join $output metrics.txt] w]
    puts $f "LATCHES [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]]"
    puts $f "BLACKBOXES [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]"
    foreach c {transport_clk memory_clk} {
        set clock [get_clocks $c]
        puts $f "PERIOD_$c [get_property PERIOD $clock]"
        foreach {kind delay} {setup max hold min} {
            report_timing -from $clock -to $clock -delay_type $delay -max_paths 10 -file [file join $output ${c}_${kind}.rpt]
            puts $f "${kind}_$c [get_property SLACK [get_timing_paths -from $clock -to $clock -delay_type $delay -max_paths 1]]"
        }
    };close $f
    set f [open [file join $output async_registers.tsv] w]
    foreach c [lsort [get_cells -hier -filter {ASYNC_REG == TRUE}]] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property ASYNC_REG $c]"
    };close $f
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_timing -from [get_ports init_complete_async*] -max_paths 10 -file [file join $output external_status.rpt]
    report_timing -from [get_clocks memory_clk] -to [get_clocks transport_clk] -max_paths 10 -file [file join $output status_crossing.rpt]
    report_drc -file [file join $output drc.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    if {$stage ne "synth"} {report_clock_utilization -file [file join $output clock_utilization.rpt]}
    if {$stage eq "route"} {report_route_status -file [file join $output route_status.rpt]}
    write_xdc -constraints all [file join $output constraints.xdc]
    if {$stage eq "route"} {write_verilog -mode funcsim [file join $output physical_funcsim.v]}
    close_design
    puts "NTRPU_HBM_PHYSICAL_COMPLETE $stage"
}
if {[llength $argv]!=3} {error "Expected output stage checkpoint"}
if {[catch {physical $root [lindex $argv 0] [lindex $argv 1] [lindex $argv 2]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
