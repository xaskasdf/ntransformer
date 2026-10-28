# Parameterized xelab tops use escaped scope names; discover the actual object.
set base {}
foreach scope [get_scopes -r /*] {
    if {[regexp {^/[^/]+$} $scope] && [string match {*tb_ntrpu_dual_hbm*} $scope]} {
        current_scope $scope
        if {[llength [get_objects -quiet total_cmds]]==1} {
            if {$base ne {}} {error "Ambiguous testbench scope"}
            set base $scope
        }
    }
}
if {$base eq {}} {error "Native testbench scope not found"}
current_scope $base
puts "DH_SCOPE $base"
set probes {}
foreach name {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n started scenario epoch fault drained inflight banks_used pending_counts total_cmds total_bytes native_read_beats DRAM_0_STAT_CATTRIP DRAM_1_STAT_CATTRIP} {
    set object [get_objects -quiet $name]
    if {[llength $object]!=1} {error "Missing public observation $name"}
    dict set probes $name $object
}
puts "DH_PROBES [dict size $probes]"
set start_wall [clock milliseconds]
set f [open progress.tsv w]
puts $f [join [concat {simulation_time wall_ms} [dict keys $probes]] "\t"]
set observed 0
for {set chunk 0} {$chunk<55} {incr chunk} {
    set before [current_time -s]
    run 1 us
    set row [list [current_time -s] [expr {[clock milliseconds]-$start_wall}]]
    dict for {name object} $probes {lappend row [get_value -radix hex $object]}
    puts $f [join $row "\t"];flush $f
    puts "DH_OBSERVE [join $row { }]";flush stdout
    if {[get_value -radix hex [dict get $probes started]] eq "1"} {
        if {[get_value -radix hex [dict get $probes apb_complete_0]] ne "1" || [get_value -radix hex [dict get $probes apb_complete_1]] ne "1"} {error "Started without both stacks ready"}
        set observed 1;break
    }
    if {[current_time -s] eq $before || [clock milliseconds]-$start_wall>1150000} {break}
}
close $f
if {!$observed} {error "Native startup window incomplete"}
puts DUAL_HBM_NATIVE_STARTUP_OBSERVED
quit
