# Use the verified post-opt checkpoint and the device-audited AXI bank plan.
if {[lindex $argv 1] eq "netlist"} {
    source [file join [file dirname [info script]] ntrpu_overlap_axi.tcl]
    exit 0
}
if {[lindex $argv 1] ni {place route}} {error "Expected place, route or netlist"}
set f [open [file join [file dirname [info script]] ntrpu_overlap_bankplan.tcl] r]
set script [read $f]; close $f
set start [string first {proc save_bank_members} $script]
set end [string first {set f [open [file join [file dirname [info script]] ntrpu_transport_study.tcl] r]} $script]
if {$start < 0 || $end <= $start} {error "Bank procedures changed"}
eval [string range $script $start [expr {$end-1}]]
proc save_bank_sites {output} {
    set f [open [file join $output bank_sites.tsv] w]; puts $f "bank\tsite"
    for {set bank 0} {$bank < 8} {incr bank} {
        foreach site [lsort [get_sites -of_objects [get_pblocks ntrpu_bank_$bank] -filter {SITE_TYPE == SLICEL || SITE_TYPE == SLICEM}]] {
            puts $f "$bank\t$site"
        }
    }
    close $f
}
set f [open [file join [file dirname [info script]] ntrpu_transport_study.tcl] r]
set script [read $f]; close $f
set needle {opt_design;place_design}
if {[string first $needle $script] < 0} {error "Placement flow changed"}
# The input is the immutable checkpoint exported immediately after opt_design.
# Do not optimize again after proving the selected dependency cones.
set script [string map [list $needle {add_bank_regions $output;place_design}] $script]
set needle {        write_checkpoint [file join $output physical_${stage}.dcp]}
if {[string first $needle $script] < 0} {error "Checkpoint flow changed"}
set script [string map [list $needle "$needle\n        report_bank_regions \$output\n        save_bank_sites \$output"] $script]
eval $script
