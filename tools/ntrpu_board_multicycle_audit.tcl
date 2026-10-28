# Diagnostic-only tighter setup analysis, never export an altered checkpoint.
proc ntrpu_multicycle_audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected ROUTED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    open_checkpoint $checkpoint
    if {[llength [get_ports]] != 221} { error "Unexpected joint checkpoint" }
    report_methodology -file [file join $output methodology.rpt]
    set pairs {}
    foreach violation [get_methodology_violations] {
        if {[get_property CHECK $violation] ne "TIMING-16"} { continue }
        set description [get_property DESCRIPTION $violation]
        if {![regexp {violation of ([-0-9.]+) ns between (.+) \(clocked by (.+)\) and (.+) \(clocked by (.+)\)\. Large} $description -> slack from from_clock to to_clock]} {
            error "Cannot parse TIMING-16 endpoints"
        }
        lappend pairs [list [get_property NAME $violation] $slack $from $to $from_clock $to_clock]
    }
    if {[llength $pairs] != 55} { error "Expected 55 reproducible diagnostics" }
    set stream [open [file join $output path_comparison.tsv] {WRONLY CREAT EXCL}]
    puts $stream "rule\tmethodology_slack_ns\tactive_slack_ns\tactive_requirement_ns\tfrom\tto"
    set i 0
    foreach pair $pairs {
        lassign $pair rule reported from to from_clock to_clock
        set fp [get_pins $from]
        set tp [get_pins $to]
        if {[llength $fp] != 1 || [llength $tp] != 1} { error "Nonunique endpoint" }
        set paths [get_timing_paths -from $fp -to $tp -delay_type max -max_paths 1]
        if {[llength $paths] != 1} { error "Missing active timing path" }
        puts $stream [join [list $rule $reported [get_property SLACK $paths] [get_property REQUIREMENT $paths] $from $to] "\t"]
        report_timing -from $fp -to $tp -delay_type max -max_paths 1 -file [file join $output path_[incr i].rpt]
    }
    close $stream
    # Controlled diagnostic on ONE exact pair: tighten setup by one source
    # cycle. This is not a fix, and is never saved as a candidate design.
    lassign [lindex $pairs 0] rule reported from to
    write_xdc [file join $output original_constraints.xdc]
    set_multicycle_path 1 -setup -start -from [get_pins $from] -to [get_pins $to]
    report_timing -from [get_pins $from] -to [get_pins $to] -delay_type max -max_paths 1 -file [file join $output tightened_diagnostic.rpt]
    write_xdc [file join $output diagnostic_only_constraints.xdc]
    puts "NTRPU_MULTICYCLE_AUDIT_EXPORTED; final in-memory setup tightened on one pair; no checkpoint saved"
    close_project
}
if {[catch {ntrpu_multicycle_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
