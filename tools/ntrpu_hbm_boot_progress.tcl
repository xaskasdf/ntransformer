# Observe an immutable cached snapshot. Never force a signal or bypass init.
set start_wall [clock milliseconds]
set log [open progress.tsv w]
puts $log "simulation_time\twall_ms\tapb_complete_0\tapb_complete_1\tmem_rst_n\tapb_rst_n"
set probes {}
foreach signal {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n} {
    set objects [get_objects -quiet /tb_ntrpu_hbm_boot/$signal]
    dict set probes $signal $objects
    puts "PROBE_OBJECT $signal [llength $objects]"
}
set inventory [open visibility.txt w]
puts $inventory "SCOPES [get_scopes -quiet /tb_ntrpu_hbm_boot/*]"
puts $inventory "TOP_OBJECTS [get_objects -quiet /tb_ntrpu_hbm_boot/*]"
close $inventory
proc sample {log start_wall probes} {
    set row [list [current_time -s] [expr {[clock milliseconds]-$start_wall}]]
    dict for {name objects} $probes {
        if {[llength $objects]==1 && ![catch {get_value -radix bin $objects} value]} {
            lappend row $value
        } else {lappend row UNAVAILABLE}
    }
    puts $log [join $row "\t"];flush $log
    puts "PROBE_PROGRESS [join $row { }]";flush stdout
}
sample $log $start_wall $probes
for {set chunk 0} {$chunk<100} {incr chunk} {
    run 1 us
    sample $log $start_wall $probes
    if {[clock milliseconds]-$start_wall>240000} {break}
}
close $log
puts "PROBE_OBSERVATION_COMPLETE"
quit
