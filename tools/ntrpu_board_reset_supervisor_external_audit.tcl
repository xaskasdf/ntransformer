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
    # Audit the external reset net directly; do not waive its CDC report.
    set f [open [file join $output management_reset_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tpreset_ports\tpreset_drivers\tdata_driver\tclock"
    for {set index 0} {$index < 3} {incr index} {
        set name [format {management_reset/arststages_ff_reg[%d]} $index]
        set cell [get_cells -quiet $name]
        if {[llength $cell] != 1} {error "Missing external reset stage"}
        set ref [get_property REF_NAME $cell]
        set init [get_property INIT $cell]
        set attribute [get_property ASYNC_REG $cell]
        set net [get_nets -segments -of_objects [get_pins $name/PRE]]
        set ports [get_ports -quiet -of_objects $net]
        set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        set dn [get_nets -segments -of_objects [get_pins $name/D]]
        set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
        set clocks [get_clocks -of_objects [get_pins $name/C]]
        puts $f "$name\t$ref\t$init\t$attribute\t$ports\t$drivers\t$dd\t$clocks"
        if {$ref ne "FDPE" || $init ne "1'b0" || $attribute ni {1 TRUE true} ||
            $ports ne "external_reset_async" || [llength $drivers] || $clocks ne "mgmt_clk"} {
            error "External reset must directly drive three marked FDPEs without a LUT"
        }
        if {$index > 0} {
            set expected [format {management_reset/arststages_ff_reg[%d]/Q} [expr {$index-1}]]
            if {$dd ne $expected} {error "Broken reset shift chain"}
        } else {
            if {[llength $dd] != 1 || [get_property REF_NAME [get_cells -of_objects $dd]] ne "GND"} {
                error "First reset stage must shift in zero"
            }
        }
    }
    close $f
    set f [open [file join $output primitive_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "ref_name\tcount"
    set counts [dict create]
    foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}] {
        set ref [get_property REF_NAME $cell]
        dict incr counts $ref
        if {[string match RAM* $ref] || [string match SRL* $ref]} {error "Unexpected RAM/SRL primitive"}
    }
    dict for {ref count} $counts {puts $f "$ref\t$count"}
    close $f
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc_all.rpt]
    write_xdc -constraints all [file join $output audit_constraints.xdc]
    puts NTRPU_SUPERVISOR_EXTERNAL_STRUCTURE_PASS
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
