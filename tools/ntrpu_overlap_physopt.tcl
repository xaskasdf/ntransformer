# Optimize a copy of the routed compact checkpoint, preserving the clock and
# all timing budgets. Explicit switches exclude clock optimization/retiming.
if {[lindex $argv 1] eq "netlist"} {
    source [file join [file dirname [info script]] ntrpu_overlap_physical.tcl]
    exit 0
}
if {[lindex $argv 1] ne "route"} {error "Expected route or netlist stage"}
proc save_primitive_state {path} {
    set cells [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]]
    set f [open $path w];puts $f "cell\tref\tloc\tbel\tmodifications"
    foreach c $cells ref [get_property REF_NAME $cells] loc [get_property LOC $cells] bel [get_property BEL $cells] mods [get_property PHYS_OPT_MODIFIED $cells] {
        puts $f "$c\t$ref\t$loc\t$bel\t$mods"
    }
    close $f
}
set f [open [file join [file dirname [info script]] ntrpu_transport_study.tcl] r]
set script [read $f];close $f
set needle {if {$stage eq "place"} {set prior synth} else {set prior place}}
if {[string first $needle $script]<0} {error "Reference checkpoint flow changed"}
set script [string map [list $needle {set prior route}] $script]
set needle {set failed [catch {route_design} message options]}
if {[string first $needle $script]<0} {error "Reference routing flow changed"}
set replacement {
                help phys_opt_design
                report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output pre_timing.rpt]
                report_utilization -file [file join $output pre_utilization.rpt]
                save_primitive_state [file join $output cells_before.tsv]
                set f [open [file join $output optimization.txt] w]
                puts $f "COMMAND phys_opt_design -critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix"
                puts $f "RETIME 0\nCLOCK_OPT 0\nFORCED_REPLICATION 0\nRTL_CHANGED 0"
                close $f
                set failed [catch {
                    phys_opt_design -critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix
                } message options]
                save_primitive_state [file join $output cells_after.tsv]
}
set script [string map [list $needle $replacement] $script]
eval $script
