# Private post-route experiment. Preserve periods, CDC constraints and clock trees.
set_param general.maxThreads 2
if {[catch {
    lassign $argv donor out strategy
    if {[llength $argv]!=3 || [file exists $out] || $strategy ni {replicate restructure explore}} {
        error "DONOR NEW_OUTPUT replicate|restructure|explore required"
    }
    file mkdir $out
    open_checkpoint $donor
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_clocked"} {error "Wrong checkpoint"}
    write_xdc -constraints all $out/before.xdc
    report_timing_summary -delay_type min_max -file $out/before_timing.rpt
    report_utilization -file $out/before_utilization.rpt
    report_timing -from [get_clocks mem_clk] -to [get_clocks mem_clk] -delay_type max -max_paths 8 -file $out/before_mem_setup.rpt
    if {$strategy eq "replicate"} {
        phys_opt_design -critical_cell_opt -path_groups mem_clk
    } elseif {$strategy eq "restructure"} {
        phys_opt_design -restruct_opt -path_groups mem_clk
    } else {
        phys_opt_design -directive AggressiveExplore
    }
    write_checkpoint $out/refined.dcp
    write_xdc -constraints all $out/after.xdc
    report_timing_summary -delay_type min_max -file $out/after_timing.rpt
    report_utilization -file $out/after_utilization.rpt
    report_timing -from [get_clocks mem_clk] -to [get_clocks mem_clk] -delay_type max -max_paths 8 -file $out/after_mem_setup.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_MEMORY_TIMING_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
