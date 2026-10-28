# Simulation-only adapter: reuse the frozen flow and its complete comparisons.
if {[llength $argv]!=2 || [lindex $argv 1] ne "netlist"} {error "Netlist stage only"}
set file [open [file join [file dirname [info script]] ntrpu_return_cut.tcl] r]
set frozen [read $file];close $file
foreach {old new} {
    {prepare_ntrpu_return_cut.py} {prepare_ntrpu_return_cut_netlist.py}
    {add_files [file join $output .. inputs physical_funcsim.v]} {add_files [file join $generated route_isolated.v]}
} {
    set first [string first $old $frozen]
    if {$first<0 || [string first $old $frozen [expr {$first+1}]]>=0} {error "Frozen boundary changed"}
    set frozen [string replace $frozen $first [expr {$first+[string length $old]-1}] $new]
}
eval $frozen
