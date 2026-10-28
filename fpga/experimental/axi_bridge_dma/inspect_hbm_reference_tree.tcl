# Read-only diagnostic of the saved native synthesis; no implementation stage.
set_param general.maxThreads 2
if {[catch {
    lassign $argv checkpoint out
    if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
    file mkdir $out
    open_checkpoint $checkpoint
    set f [open $out/reference_connections.tsv {WRONLY CREAT EXCL}]
    puts $f "query\tpin\tdirection\tprimitive"
    foreach i {0 1} {
        foreach query [list board_inputs/hbm_reference${i}_input/O hbm_reference${i}/I hbm_reference${i}/O hbm_binding/memory/HBM_REF_CLK_${i}] {
            set pin [get_pins $query]
            set nets [get_nets -segments -of_objects $pin]
            foreach leaf [lsort [get_pins -leaf -of_objects $nets]] {
                puts $f "$query\t$leaf\t[get_property DIRECTION [get_pins $leaf]]\t[get_property REF_NAME [get_cells -of_objects [get_pins $leaf]]]"
            }
        }
    };close $f
    report_clocks -file $out/clocks.rpt
    report_methodology -file $out/methodology.rpt
    report_clock_utilization -file $out/clock_utilization.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_REFERENCE_DIAGNOSTIC_EXPORTED
    close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
