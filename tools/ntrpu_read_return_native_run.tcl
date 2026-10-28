# Parameterized xelab tops use escaped scope names; discover the actual object.
set base {}
foreach scope [get_scopes -r /*] {
    if {[regexp {^/[^/]+$} $scope] && [string match {*tb_ntrpu_read_return*} $scope]} {
        current_scope $scope
        if {[llength [get_objects -quiet commands]]==1} {
            if {$base ne {}} {error "Ambiguous testbench scope"}
            set base $scope
        }
    }
}
if {$base eq {}} {error "Native testbench scope not found"}
current_scope $base
puts "RR_SCOPE $base"
set probes {}
foreach name {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n started scenario epoch fault drained stopped commands read_beats output_beats completions DRAM_0_STAT_CATTRIP DRAM_1_STAT_CATTRIP} {
    set object [get_objects -quiet $name]
    if {[llength $object]!=1} {error "Missing public observation $name"}
    dict set probes $name $object
}
puts "RR_PROBES [dict size $probes]"
set start_wall [clock milliseconds]
set f [open progress.tsv w]
puts $f [join [concat {simulation_time wall_ms} [dict keys $probes]] "\t"]
for {set chunk 0} {$chunk<120} {incr chunk} {
    set before [current_time -s]
    run 1 us
    set row [list [current_time -s] [expr {[clock milliseconds]-$start_wall}]]
    dict for {name object} $probes {lappend row [get_value -radix hex $object]}
    puts $f [join $row "\t"];flush $f
    puts "RR_PROGRESS [join $row { }]";flush stdout
    if {[current_time -s] eq $before || [clock milliseconds]-$start_wall>1750000} {break}
}
close $f
puts NTRPU_READ_RETURN_NATIVE_WINDOW_COMPLETE
quit
