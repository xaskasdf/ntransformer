# One bounded post-route experiment; never edits the sealed routed donor.
set_param general.maxThreads 2
source [file join [file dirname [info script]] implement_hbm_board.tcl]

proc ho_run {checkpoint out mode} {
    if {[file exists $out] || $mode ni {explore reopen}} {error "Fresh output and supported mode required"}
    set source [file dirname [info script]]
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong board checkpoint"}
    write_xdc -constraints all $out/before.xdc
    bi_pins $out 0
    report_route_status -file $out/route_before.rpt
    report_timing_summary -delay_type min_max -file $out/timing_before.rpt
    set f [open $out/experiment.tsv {WRONLY CREAT EXCL}]
    puts $f "mode\tcommand\tcpu_limit\tclock_or_exception_edits\tqualified"
    if {$mode eq "explore"} {
        puts $f "explore\tphys_opt_design -directive Explore\t2\t0\t0";close $f
        phys_opt_design -directive Explore
        set stage postroute_explore
        write_checkpoint $out/$stage.dcp
    } else {
        puts $f "reopen\topen_checkpoint\t2\t0\t0";close $f
        set stage reopened
    }
    bi_reports $out/reports
    bi_audit $source $out/contracts
    if {$mode eq "reopen"} {
        source $source/reconcile_hbm_pcie_timing.tcl
        hp_reconcile $source $out/pcie_reconciliation
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f BOARD_IMPLEMENTATION_${stage}_EXPORTED_REVIEW_REQUIRED;close $f
    close_design
}
