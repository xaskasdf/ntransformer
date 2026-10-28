# One targeted post-route pass; no timing constraints are added or relaxed.
set_param general.maxThreads 2
if {[catch {
    lassign $argv donor out
    if {[llength $argv]!=2 || [file exists $out]} {error "DONOR NEW_OUTPUT required"}
    file mkdir $out
    open_checkpoint $donor
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong device"}
    write_xdc -constraints all $out/before.xdc
    report_timing_summary -delay_type min_max -file $out/before_timing.rpt
    phys_opt_design -placement_opt -routing_opt -critical_pin_opt -path_groups mem_clk
    write_checkpoint $out/refined.dcp
    write_xdc -constraints all $out/after.xdc
    report_timing_summary -delay_type min_max -file $out/after_timing.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_RANGE_REFINEMENT_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
