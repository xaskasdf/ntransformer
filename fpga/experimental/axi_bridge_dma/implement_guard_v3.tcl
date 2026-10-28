# Exploratory PCIe-only OOC implementation. Not a released PCB pinout.
set_param general.maxThreads 2
source [file join [file dirname [info script]] audit_guard_v3.tcl]
proc report_stage {out} {
    file mkdir $out
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_drc -file $out/drc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_clock_utilization -file $out/clock_utilization.rpt
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_timing -from [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && IS_PRIMITIVE == 1}] -to [get_cells -hierarchical -filter {NAME =~ guarded/client/* && IS_PRIMITIVE == 1}] -delay_type min_max -max_paths 20 -file $out/guard_to_client.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    write_xdc -constraints all $out/effective.xdc
}
if {[catch {
    if {[llength $argv]!=3} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv dcp out reference
    if {[file exists $out]} {error "New output required"}
    file mkdir $out
    v3_reference_alias $reference $out/reference_alias.tsv
    open_checkpoint $dcp
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Unexpected part"}
    write_xdc -constraints all $out/input.xdc
    set f [open $out/candidate_pins.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tball\tfunction\tbank\tstatus"
    foreach {port ball function bank} {
        sys_clk_p AR40 MGTREFCLK0P_125 125
        sys_clk_n AR41 MGTREFCLK0N_125 125
        sys_rst_n BF41 IO_T3U_N12_PERSTN0_65 65
    } {
        set p [get_package_pins $ball]
        if {[llength $p]!=1 || [get_property PIN_FUNC $p] ne $function || [get_property BANK $p] ne $bank || ![get_property IS_BONDED $p]} {error "Package database disagrees with candidate"}
        set_property PACKAGE_PIN $ball [get_ports $port]
        puts $f "$port\t$ball\t$function\t$bank\tEXPLORATORY_PCIE_ONLY"
    }
    close $f
    set_property IOSTANDARD LVCMOS18 [get_ports sys_rst_n]
    # Fabric inputs, management clock and cold configuration reset remain OOC
    # contracts. No zero I/O delays, clock-source fiction, broad cuts or waivers.
    write_xdc -constraints all $out/candidate.xdc
    write_checkpoint $out/candidate.dcp
    # Input is already optimized and audited for shared native resets.
    write_checkpoint $out/optimized.dcp
    place_design
    write_checkpoint $out/placed.dcp
    report_stage $out/placed
    route_design
    write_checkpoint $out/routed.dcp
    report_route_status -file $out/route_status.rpt
    report_stage $out/routed
    v3_evidence $out/v3_audit
    set f [open $out/guard_buffers.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tdriver\tce_driver\tce_ref\tclock_loads\tdata_loads\treset_loads"
    foreach c [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && REF_NAME =~ BUFG*}] {
        set input [get_pins -of_objects $c -filter {REF_PIN_NAME == I}]
        set ce [get_pins -of_objects $c -filter {REF_PIN_NAME == CE}]
        set driver [get_pins -leaf -of_objects [get_nets -segments -of_objects $input] -filter {DIRECTION == OUT}]
        set enable [get_pins -leaf -of_objects [get_nets -segments -of_objects $ce] -filter {DIRECTION == OUT}]
        set nc 0;set nd 0;set nr 0
        foreach p [all_fanout -flat -endpoints_only -from [get_pins -of_objects $c -filter {DIRECTION == OUT}]] {
            if {[get_property CLASS $p] ne "pin"} {continue}
            if {[get_property IS_CLOCK $p]} {incr nc} elseif {[get_property REF_PIN_NAME $p] in {R S CLR PRE}} {incr nr} else {incr nd}
        }
        puts $f "$c\t[get_property REF_NAME $c]\t$driver\t$enable\t[get_property REF_NAME [get_cells -of_objects $enable]]\t$nc\t$nd\t$nr"
        if {$nc!=0} {error "Guard control buffer drives clock inputs"}
    }
    close $f
    report_methodology -file $out/routed/methodology.rpt
    report_waivers -file $out/routed/waivers.rpt
    set f [open $out/guard_placement.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tasync_reg\tloc\tbel"
    foreach c [lsort [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && IS_PRIMITIVE == 1}]] {
        puts $f [join [list $c [get_property REF_NAME $c] [get_property ASYNC_REG $c] [get_property LOC $c] [get_property BEL $c]] "\t"]
    }
    close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f V3_SHARED_NATIVE_ROUTED_REVIEW_REQUIRED
    close $f
    close_design
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
