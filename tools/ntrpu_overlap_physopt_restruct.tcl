# Second bounded pass: permit LUT restructuring, without sequential retiming.
set f [open [file join [file dirname [info script]] ntrpu_overlap_physopt.tcl] r]
set script [read $f];close $f
set needle {phys_opt_design -critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix}
if {[llength [split [string map [list $needle \u0001] $script] \u0001]]!=3} {error "Expected recorded and executed optimization commands"}
set script [string map [list $needle {phys_opt_design -restruct_opt -critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix}] $script]
eval $script
