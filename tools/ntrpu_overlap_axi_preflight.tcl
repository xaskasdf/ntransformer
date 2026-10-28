# Inspect the exact post-opt_design point without changing the source checkpoint.
set_param general.maxThreads 2
if {[llength $argv] != 3} {error "Expected checkpoint, output, and candidate plan"}
lassign $argv checkpoint output plan
file mkdir $output
if {[catch {
    open_checkpoint $checkpoint
    write_xdc -constraints all [file join $output constraints_before.xdc]
    opt_design
    # Reuse the established leaf-pin exporter, without opening/closing another design.
    set f [open [file join [file dirname [info script]] ntrpu_overlap_connectivity.tcl] r]
    set script [read $f]; close $f
    set start [string first {    set cells [lsort} $script]
    set end [string first {    close_design} $script]
    if {$start < 0 || $end <= $start} {error "Connectivity exporter changed"}
    eval [string range $script $start [expr {$end-1}]]
    set f [open [file join $output device_sites.tsv] w]; puts $f "site\ttype"
    set sites [lsort [get_sites -filter {SITE_TYPE == SLICEL || SITE_TYPE == SLICEM}]]
    foreach s $sites t [get_property SITE_TYPE $sites] {puts $f "$s\t$t"}; close $f
    if {[llength [get_pblocks -quiet]]} {error "Unexpected existing Pblocks"}
    set f [open [file join $plan groups.tsv] r]; set lines [split [string trim [read $f]] "\n"]; close $f
    if {[lindex $lines 0] ne "bank\tcell\trange"} {error "Plan schema changed"}
    set ranges [dict create]
    foreach line [lrange $lines 1 end] {
        lassign [split $line "\t"] bank cell range
        if {[dict exists $ranges $bank] && [dict get $ranges $bank] ne $range} {error "Inconsistent range"}
        dict set ranges $bank $range
    }
    if {[dict size $ranges] != 8} {error "Expected eight regions"}
    set f [open [file join $output geometry.tsv] w]; puts $f "bank\trequested\tapplied\tsites"
    for {set bank 0} {$bank < 8} {incr bank} {
        set pb [create_pblock preflight_bank_$bank]
        resize_pblock $pb -add [dict get $ranges $bank]
        set actual [lsort [get_sites -of_objects $pb -filter {SITE_TYPE == SLICEL || SITE_TYPE == SLICEM}]]
        puts $f "$bank\t[dict get $ranges $bank]\t[get_property GRID_RANGES $pb]\t[join $actual {;}]"
        delete_pblocks $pb
    }
    close $f
    write_xdc -constraints all [file join $output constraints_after.xdc]
    write_checkpoint [file join $output post_opt.dcp]
    close_design
    puts "NTRPU_AXI_PREFLIGHT_COMPLETE"
} message options]} {
    puts stderr $message; puts stderr [dict get $options -errorinfo]; exit 1
}
exit 0
