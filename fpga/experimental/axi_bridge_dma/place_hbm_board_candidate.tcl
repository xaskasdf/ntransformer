# Exploratory OOC placement only. Four existing pads; six fabric boundaries remain.
set_param general.maxThreads 2
proc bp_one {xs label} {
    if {[llength $xs]!=1} {error "$label: expected one object, got $xs"}
    return [lindex $xs 0]
}
proc bp_reports {out} {
    file mkdir $out
    report_drc -file $out/drc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_clock_utilization -file $out/clock_utilization.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    write_xdc -constraints all $out/effective.xdc
}
if {[catch {
    if {[llength $argv]!=2} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv checkpoint out
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled"} {error "Unexpected composition"}
    write_xdc -constraints all $out/before.xdc
    # Check the donor before touching either half of the differential pair:
    # assigning its positive package pin also resolves the negative pin.
    foreach port {sys_clk_p sys_clk_n sys_rst_n fabric_ref_clk_in} {
        if {[get_property PACKAGE_PIN [get_ports $port]] ne ""} {error "Donor already assigns $port"}
    }
    set f [open $out/candidate_pins.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tball\tfunction\tbank\tstandard\tstatus"
    foreach {port ball function bank standard} {
        sys_clk_p AR40 MGTREFCLK0P_125 125 NONE
        sys_clk_n AR41 MGTREFCLK0N_125 125 NONE
        sys_rst_n BF41 IO_T3U_N12_PERSTN0_65 65 LVCMOS18
        fabric_ref_clk_in BK31 IO_L11P_T1U_N8_GC_64 64 LVCMOS18
    } {
        set p [bp_one [get_package_pins $ball] package_pin]
        if {[get_property PIN_FUNC $p] ne $function || [get_property BANK $p] ne $bank || ![get_property IS_BONDED $p]} {error "Candidate package mismatch: $ball"}
        set current [get_property PACKAGE_PIN [get_ports $port]]
        if {$current ne "" && $current ne $ball} {error "Conflicting inferred package pin for $port: $current"}
        set_property PACKAGE_PIN $ball [get_ports $port]
        if {$standard ne "NONE"} {set_property IOSTANDARD $standard [get_ports $port]}
        puts $f "$port\t$ball\t$function\t$bank\t$standard\tEXPERIMENT_ONLY";
    };close $f
    if {[get_property DIFF_PAIR_PIN [get_package_pins AR40]] ne "AR41"} {error "PCIe reference pair changed"}
    set gc [bp_one [get_sites -of_objects [get_package_pins BK31]] fabric_gc_site]
    if {![get_property IS_GLOBAL_CLK [get_package_pins BK31]] || [get_slrs -of_objects $gc] ne "SLR0"} {error "Fabric GCIO not in candidate SLR"}
    # IBUFDS_GTE4 is a BEL within a GTYE4_COMMON site, not a site type.
    set refsite [bp_one [get_sites -of_objects [get_package_pins AR40] -filter {SITE_TYPE == GTYE4_COMMON}] pcie_reference_site]
    if {$refsite ne "GTYE4_COMMON_X0Y1"} {error "PCIe reference site changed: $refsite"}
    set_property LOC $refsite [get_cells refclk_ibuf]
    set f [open $out/reference_site.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tsite\tball\tstatus";puts $f "refclk_ibuf\t$refsite\tAR40\tEXPERIMENT_ONLY";close $f
    write_xdc -constraints all $out/candidate.xdc
    write_checkpoint $out/candidate.dcp
    # No input delays, clock cuts, waivers, clock-routing overrides or bitstream.
    bp_reports $out/pinned
    opt_design
    write_checkpoint $out/optimized.dcp
    bp_reports $out/optimized
    place_design
    write_checkpoint $out/placed.dcp
    bp_reports $out/placed
    report_methodology -file $out/placed/methodology.rpt
    report_waivers -file $out/placed/waivers.rpt
    set f [open $out/readiness_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\trequirement_ns\tslack_ns"
    foreach from {availability/fault_reg/C {availability/ready_mgmt_reg[0]/C} {availability/ready_mgmt_reg[1]/C}} {
        foreach type {max min} {
            set p [bp_one [get_timing_paths -quiet -from [get_pins $from] -to [get_pins loader_readiness/ready_reg/D] -delay_type $type -max_paths 1] readiness_path]
            puts $f "$from\tloader_readiness/ready_reg/D\t$type\t[get_property REQUIREMENT $p]\t[get_property SLACK $p]"
        }
    };close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_EXPLORATORY_PLACEMENT_REVIEW_REQUIRED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
