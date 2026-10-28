# Read-only structural audit of the isolated supervisor synthesis checkpoint.
proc supervisor_audit {arguments} {
    if {[llength $arguments] != 2} {error "Expected SYNTH_DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_reset_supervisor" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 3} {
        error "Unexpected design context"
    }
    set f [open [file join $output release_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg"
    set total 0
    foreach name {host_reset_cdc device_reset_cdc host_admit_cdc device_admit_cdc} {
        set cells [get_cells -quiet -hierarchical -filter "NAME =~ $name/release_ff_reg*"]
        if {[llength $cells] != 3} {error "Expected three release stages: $name"}
        foreach cell [lsort $cells] {
            set ref [get_property REF_NAME $cell]
            set init [get_property INIT $cell]
            set attribute [get_property ASYNC_REG $cell]
            puts $f "$cell\t$ref\t$init\t$attribute"
            if {$ref ne "FDCE" || $init ne "1'b0" || $attribute ni {1 TRUE true}} {
                error "Reset release must be an INIT0 asynchronous-clear synchronizer: $cell"
            }
            incr total
        }
    }
    close $f
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
    set boxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
    if {[llength $latches] || [llength $boxes]} {error "Transparent storage or blackbox"}
    report_cdc -details -file [file join $output cdc.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_clocks -file [file join $output clocks.rpt]
    puts "NTRPU_SUPERVISOR_INIT_AUDIT_PASS registers=$total"
    close_project
}
if {[catch {supervisor_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
