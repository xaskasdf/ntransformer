# Read-only inventory of the exact reset chains in a reopened synthesis R2 DCP.
# This audit must not create any timing exception or write a checkpoint.
proc ntrpu_retained_start_reset_inventory_r2 {prefix} {
    set cells_by_name [dict create]
    foreach cell [get_cells -hierarchical] {dict set cells_by_name $cell $cell}
    set chains {admission_cdc source_count_cdc source_held_cdc source_reset_cdc source_up_cdc
        memory[0].count_cdc memory[0].held_cdc memory[0].ready_cdc memory[0].reset_cdc memory[0].up_cdc
        memory[1].count_cdc memory[1].held_cdc memory[1].ready_cdc memory[1].reset_cdc memory[1].up_cdc}
    set clears {};set data_paths {}
    foreach chain $chains {
        set previous {};set clock {};set clear_nets {}
        for {set i 0} {$i<3} {incr i} {
            set name [format {%s%s/release_ff_reg[%d]} $prefix $chain $i]
            if {![dict exists $cells_by_name $name]} {error "Missing exact reset stage $name"}
            set cell [dict get $cells_by_name $name]
            if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0" || ![get_property ASYNC_REG $cell]} {error "Unexpected reset primitive $cell"}
            set cp [get_pins -of $cell -filter {REF_PIN_NAME==C}]
            set dp [get_pins -of $cell -filter {REF_PIN_NAME==D}]
            set rp [get_pins -of $cell -filter {REF_PIN_NAME==CLR}]
            if {[llength $cp]!=1 || [llength $dp]!=1 || [llength $rp]!=1} {error "Missing reset pins $cell"}
            set clocks [get_clocks -of_objects $cp]
            set nets [lsort [get_nets -segments -of $rp]]
            if {[llength $clocks]!=1 || ![llength $nets]} {error "Ambiguous reset clock/net $cell"}
            if {$i==0} {set clock $clocks;set clear_nets $nets}
            if {$clock ne $clocks || $clear_nets ne $nets} {error "Inconsistent chain $chain"}
            set drivers [get_pins -of [get_nets -segments -of $dp] -filter {DIRECTION==OUT}]
            if {$i==0} {
                if {[llength $drivers]!=1 || [get_property REF_NAME [get_cells -of $drivers]] ne "VCC"} {error "First stage is not constant-one $cell"}
            } else {
                set qp [get_pins -of $previous -filter {REF_PIN_NAME==Q}]
                if {$drivers ne $qp} {error "Not a direct reset release chain $cell"}
                lappend data_paths [list $previous $dp]
            }
            lappend clears $rp
            set previous $cell
        }
    }
    if {[llength $clears]!=45 || [llength $data_paths]!=30} {error "Unexpected synchronizer inventory"}
    # Return only observations; preserve all existing constraints.
    return [dict create clear_pins $clears data_paths $data_paths]
}

proc retained_start_dcp_audit_r2 {output checkpoint mode} {
    if {$mode ne "start" || [file exists $output]} {error "Expected new output and start mode"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    set inventory [ntrpu_retained_start_reset_inventory_r2 {}]
    set f [open [file join $output scoped_reset_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin"
    foreach pin [dict get $inventory clear_pins] {puts $f $pin}
    close $f
    set f [open [file join $output data_paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "endpoint\tstartpoint\tslack"
    foreach pair [dict get $inventory data_paths] {
        lassign $pair previous dp
        set path [get_timing_paths -quiet -no_report_unconstrained -from $previous -to $dp -max_paths 1]
        if {[llength $path]!=1} {error "Missing stage-to-stage D path $dp"}
        puts $f "$dp\t[get_property STARTPOINT_PIN $path]\t[get_property SLACK $path]"
    }
    close $f
    set remaining 0
    foreach source {mgmt_clk source_clk ui0_clk ui1_clk} {
        foreach dest {mgmt_clk source_clk ui0_clk ui1_clk} {
            if {$source eq $dest} {continue}
            incr remaining [llength [get_timing_paths -quiet -no_report_unconstrained -from [get_clocks $source] -to [get_clocks $dest] -max_paths 1000]]
        }
    }
    if {$remaining!=0} {error "Unresolved cross-clock timed paths $remaining"}
    report_methodology -file [file join $output methodology.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_drc -file [file join $output drc.rpt]
    report_waivers -file [file join $output waivers.rpt]
    write_xdc -constraints all [file join $output effective.xdc]
    close_design
    puts NTRPU_RETAINED_START_DCP_AUDIT_R2_EXPORTED
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_start_dcp_audit_r2 [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]] [lindex $argv 2]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
