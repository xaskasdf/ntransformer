# Bounded diagnostic fallback: legal routing, explicitly not timing optimization.
# Preserve all timing constraints and measure their violations after routing.
set root [file normalize [file join [file dirname [info script]] ..]]
proc route_probe {root out} {
    file mkdir $out
    set_param general.maxThreads 2
    set f [open $root/tools/ntrpu_hbm_dma_rev2_physical.tcl];set text [read $f];close $f
    set boundary [string first "\nif \{\[catch \{physical" $text]
    if {$boundary<0} {error "Physical helper boundary changed"}
    uplevel #0 [string range $text 0 [expr {$boundary-1}]]
    open_checkpoint $root/evidence/placed.dcp
    foreach name {interface.tsv cold_reset_endpoints.txt} {file copy $root/evidence/$name $out/$name}
    set f [open $out/ROUTE_DIRECTIVE.txt w];puts $f "Quick: diagnostic legal routing; timing constraints retained for measurement, not optimized";close $f
    set body [info body physical]
    set start [string first {    route_design -directive Explore} $body]
    if {$start<0} {error "Route/report boundary changed"}
    set tail [string range $body $start end]
    if {[regexp -all {route_design -directive Explore} $tail]!=1} {error "Ambiguous routing command"}
    set tail [string map [list {route_design -directive Explore} {route_design -directive Quick}] $tail]
    eval $tail
}
if {[catch {route_probe $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
