# Prepared read-only R12 diagnostic. Execution and review are separate gates.
# report_timing/all_registers options checked against archived Vivado 2026.1 help.
proc census_write {path text} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f $text
    close $f
}

proc inventory_path_census {checkpoint output} {
    if {[llength [get_projects -quiet]] || [file exists $output]} {error "Expected new output and no project"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach command {open_checkpoint report_timing report_timing_summary all_registers report_exceptions write_xdc} {
        set text [help $command]
        if {[string trim $text] eq ""} {error "Empty help: $command"}
        census_write [file join $output help_${command}.txt] $text
    }
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_inventory_join" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected design"}
    if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {error "Unexpected clock"}
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected external clock context"}
    set registers [all_registers -cells]
    set fdre [get_cells -hierarchical -filter {REF_NAME == FDRE}]
    if {[lsort $registers] ne [lsort $fdre] || [llength $registers] != 10729} {error "Unexpected sequential inventory"}
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Unexpected blackbox/latch"}
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
    if {[llength $buffers]} {error "Expected original synthesis DCP without diagnostic BUFG"}
    set inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set outputs [all_outputs]
    if {![llength $inputs] || ![llength $outputs]} {error "Missing public ports"}

    # These are disjoint reporting categories, never timing constraints.
    set patterns [dict create \
        producer_bytes {^producer/(remaining_bytes|byte_range_high)_reg(\[|$)} \
        producer_tags_reads {^producer/(allocated_tags|tag_count|tag_range_high|outstanding_reads|read_range_high)_reg(\[|$)} \
        producer_table {^producer/(remaining|address_next|traffic_class|attributes)_reg(\[|$)} \
        producer_context {^producer/(rc_[a-z_]+|rq_[a-z_]+|held_rc_[a-z_]+|held_rq_[a-z_]+)_reg(\[|$)} \
        producer_metadata {^producer/(canonical_delivered|metadata_data|metadata_owned)_reg(\[|$)} \
        producer_control {^producer/(observing|requests|completed_reads|posted_writes|fault_latched|range_checked|saved_requester|saved_attribute_mask)_reg(\[|$)} \
        capture {^(captured_[a-z_]+|capture_effect_pending)_reg(\[|$)} \
        public_retirement {^(allocated_tags|completed_reads|observing|outstanding_reads|posted_writes|rc_partial|remaining_bytes|requests|retired_early_faults|retired_sequence|retired_valid|rq_partial|saved_attribute_mask|saved_requester|snapshot_valid|tag_count|validation_latched)_reg(\[|$)}]
    set groups [dict create]
    foreach label [dict keys $patterns] {dict set groups $label {}}
    dict set groups unclassified {}
    set f [open [file join $output register_groups.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tgroups"
    set overlaps 0
    foreach cell [lsort $registers] {
        set name [get_property NAME $cell]
        set matches {}
        dict for {label pattern} $patterns {if {[regexp $pattern $name]} {lappend matches $label}}
        if {![llength $matches]} {set matches [list unclassified]}
        if {[llength $matches] > 1} {incr overlaps}
        foreach label $matches {dict lappend groups $label $cell}
        puts $f "$name\t[get_property REF_NAME $cell]\t[join $matches {;}]"
    }
    close $f
    census_write [file join $output classification_patterns.txt] $patterns
    census_write [file join $output classification_overlaps.txt] $overlaps
    if {$overlaps} {error "Overlapping register categories recorded; review classification"}
    set f [open [file join $output subset_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tobjects"
    puts $f "all_registers\t[llength $registers]"
    puts $f "data_inputs\t[llength $inputs]"
    puts $f "outputs\t[llength $outputs]"
    dict for {label cells} $groups {puts $f "$label\t[llength $cells]"}
    close $f

    write_xdc -constraints all [file join $output constraints_before.xdc]
    report_clocks -file [file join $output clocks.rpt]
    report_property -all [current_design] -file [file join $output design.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    report_exceptions -summary -file [file join $output exception_summary.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_drc -file [file join $output drc.rpt]
    foreach type {max min} {
        report_timing -delay_type $type -max_paths 20 -from $registers -to $registers -file [file join $output all_reg_to_reg_${type}.rpt]
        report_timing -delay_type $type -max_paths 20 -from $inputs -to $registers -file [file join $output all_input_to_reg_${type}.rpt]
        report_timing -delay_type $type -max_paths 20 -from $registers -to $outputs -file [file join $output all_reg_to_output_${type}.rpt]
        report_timing -delay_type $type -max_paths 20 -from $inputs -to $outputs -file [file join $output all_input_to_output_${type}.rpt]
        dict for {label cells} $groups {
            if {![llength $cells]} {
                census_write [file join $output ${label}_${type}_empty.txt] "No registers; no report_timing query issued."
                continue
            }
            report_timing -delay_type $type -max_paths 20 -to $cells -file [file join $output ${label}_any_to_reg_${type}.rpt]
            report_timing -delay_type $type -max_paths 20 -from $inputs -to $cells -file [file join $output ${label}_input_to_reg_${type}.rpt]
            report_timing -delay_type $type -max_paths 20 -from $registers -to $cells -file [file join $output ${label}_reg_to_reg_${type}.rpt]
        }
    }
    write_xdc -constraints all [file join $output constraints_after.xdc]
    close_project
    puts NTRPU_PCIE_INVENTORY_PATH_CENSUS_EXPORTED
}
if {[llength $argv] != 2} {exit 1}
if {[catch {inventory_path_census [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
