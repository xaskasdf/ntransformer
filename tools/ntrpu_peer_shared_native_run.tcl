set base {}
foreach scope [get_scopes -r /*] {
    if {[regexp {^/[^/]+$} $scope] && [string match {*tb_ntrpu_peer_shared_native*} $scope]} {
        if {$base ne {}} {error "Ambiguous peer bench scope"}
        set base $scope
    }
}
if {$base eq {}} {error "Peer native bench not found"}
current_scope $base
set probes {}
foreach name {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n started scenario epoch reader_fault wfault halted quarantined DRAM_0_STAT_CATTRIP DRAM_1_STAT_CATTRIP} {
    set object [get_objects -quiet $name]
    if {[llength $object]!=1} {error "Missing observation $name"}
    dict set probes $name $object
}
set started_wall [clock milliseconds]
set f [open progress.tsv w]
puts $f [join [concat {simulation_time wall_ms} [dict keys $probes]] "\t"]
for {set chunk 0} {$chunk<160} {incr chunk} {
    set before [current_time -s]
    run 1 us
    set row [list [current_time -s] [expr {[clock milliseconds]-$started_wall}]]
    dict for {name object} $probes {lappend row [get_value -radix hex $object]}
    puts $f [join $row "\t"];flush $f
    puts "PN PROGRESS [join $row { }]";flush stdout
    if {[current_time -s] eq $before || [clock milliseconds]-$started_wall>1750000} {break}
}
close $f
puts NTRPU_SHARED_NATIVE_WINDOW_COMPLETE
quit
