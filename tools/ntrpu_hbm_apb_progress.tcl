# Read-only initialization diagnostics from the cached native HBM snapshot.
set start_wall [clock milliseconds]
set probes {}
foreach name {apb_complete_0 apb_complete_1 mem_rst_n apb_rst_n} {
    dict set probes $name [get_objects -quiet /tb_ntrpu_hbm_boot/$name]
}
# XSIM preserves this generate/instance pair as one scope containing a dot.
set base /tb_ntrpu_hbm_boot/dut/memory/inst/TWO_STACK.u_hbm_top
foreach stack {0 1} {
    foreach pattern {xpm_addra_%s gen_apb_addr_%s gen_poll_%s init_seq_complete_%s apb_poll_complete_%s valid_window_fail_%s intrnl_apb_psel_%s_s intrnl_apb_penable_%s_s intrnl_apb_pwrite_%s_s intrnl_apb_paddr_%s_s intrnl_apb_pready_%s_s intrnl_apb_prdata_%s_s intrnl_apb_pslverr_%s_s} {
        set name [format $pattern $stack]
        dict set probes $name [get_objects -quiet $base/$name]
    }
}
set inventory [open visibility.txt w]
dict for {name objects} $probes {puts $inventory "$name\t[llength $objects]\t$objects"}
foreach scope [list /tb_ntrpu_hbm_boot /tb_ntrpu_hbm_boot/dut /tb_ntrpu_hbm_boot/dut/memory /tb_ntrpu_hbm_boot/dut/memory/inst /tb_ntrpu_hbm_boot/dut/memory/inst/TWO_STACK $base] {
    puts $inventory "SCOPES $scope [get_scopes -quiet $scope/*]"
    puts $inventory "OBJECTS $scope [get_objects -quiet $scope/*]"
}
close $inventory
set internal_visible 0
dict for {name objects} $probes {
    if {[string match xpm_addra_* $name] || [string match intrnl_apb_* $name]} {
        incr internal_visible [llength $objects]
    }
}
if {$internal_visible==0} {
    puts APB_INTERNAL_VISIBILITY_UNAVAILABLE
    quit
}
set log [open progress.tsv w]
puts $log [join [concat {simulation_time wall_ms} [dict keys $probes]] "\t"]
proc sample {log start_wall probes} {
    set row [list [current_time -s] [expr {[clock milliseconds]-$start_wall}]]
    dict for {name objects} $probes {
        if {[llength $objects]==1 && ![catch {get_value -radix hex $objects} value]} {
            lappend row $value
        } else {lappend row UNAVAILABLE}
    }
    puts $log [join $row "\t"];flush $log
    puts "APB_PROGRESS [join $row { }]";flush stdout
}
sample $log $start_wall $probes
for {set chunk 0} {$chunk<30} {incr chunk} {
    run 1 us
    sample $log $start_wall $probes
    if {[clock milliseconds]-$start_wall>240000} {break}
}
close $log
puts APB_OBSERVATION_COMPLETE
quit
