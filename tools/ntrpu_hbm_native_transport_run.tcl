# Native initialization is observed, never forced or bypassed.
set start_wall [clock milliseconds]
set base /tb_ntrpu_hbm_native_transport
set probes {}
foreach name {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n started booted scenario fault drained inflight banks_used queued pending_counts native_aw native_w native_b native_ar native_r native_r_stalls cpl_stalls thermal_unknown DRAM_0_STAT_CATTRIP DRAM_1_STAT_CATTRIP} {
    dict set probes $name [get_objects -quiet $base/$name]
}
foreach index {0 1} {
    dict set probes init_seq_complete_$index [get_objects -quiet $base/hbm/memory/inst/TWO_STACK.u_hbm_top/init_seq_complete_$index]
}
set f [open progress.tsv w]
puts $f [join [concat {simulation_time wall_ms} [dict keys $probes]] "\t"]
for {set chunk 0} {$chunk<80} {incr chunk} {
    set before [current_time -s]
    run 1 us
    set row [list [current_time -s] [expr {[clock milliseconds]-$start_wall}]]
    dict for {name objects} $probes {
        if {[llength $objects]==1 && ![catch {get_value -radix hex $objects} value]} {lappend row $value} else {lappend row UNAVAILABLE}
    }
    puts $f [join $row "\t"];flush $f
    puts "NC_PROGRESS [join $row { }]";flush stdout
    if {[current_time -s] eq $before || [clock milliseconds]-$start_wall>850000} {break}
}
close $f
puts NTRPU_NATIVE_TRANSPORT_WINDOW_COMPLETE
quit
