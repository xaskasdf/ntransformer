# Resume a frozen optimized OOC composition. No new constraints or netlist edits.
set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=2} {error "OPTIMIZED_CHECKPOINT NEW_OUTPUT required"}
    lassign $argv checkpoint out
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled"} {error "Unexpected composition"}
    foreach {port ball} {sys_clk_p AR40 sys_clk_n AR41 sys_rst_n BF41 fabric_ref_clk_in BK31} {
        if {[get_property PACKAGE_PIN [get_ports $port]] ne $ball} {error "Donor pin differs: $port"}
    }
    foreach port {sys_rst_n fabric_ref_clk_in} {
        if {[get_property IOSTANDARD [get_ports $port]] ne "LVCMOS18"} {error "Donor I/O standard differs"}
    }
    if {[get_property LOC [get_cells refclk_ibuf]] ne "GTYE4_COMMON_X0Y1"} {error "Donor reference site differs"}
    write_xdc -constraints all $out/before.xdc
    place_design
    # Save immediately, before potentially expensive diagnostics.
    write_checkpoint $out/placed.dcp
    write_xdc -constraints all $out/after.xdc
    report_drc -file $out/drc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_bus_skew -file $out/bus_skew.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_exceptions -file $out/exceptions.rpt
    report_clock_utilization -file $out/clock_utilization.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_methodology -file $out/methodology.rpt
    report_waivers -file $out/waivers.rpt
    set f [open $out/readiness_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\trequirement_ns\tslack_ns"
    foreach from {availability/fault_reg/C {availability/ready_mgmt_reg[0]/C} {availability/ready_mgmt_reg[1]/C}} {
        foreach type {max min} {
            set p [get_timing_paths -quiet -from [get_pins $from] -to [get_pins loader_readiness/ready_reg/D] -delay_type $type -max_paths 1]
            if {[llength $p]!=1} {error "Readiness path missing: $from $type"}
            puts $f "$from\tloader_readiness/ready_reg/D\t$type\t[get_property REQUIREMENT $p]\t[get_property SLACK $p]"
        }
    };close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_BOARD_PLACED_REVIEW_REQUIRED
    close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
