# Read-only structural and timing evidence for the shared native profile.
proc v3_pin {name} {
    set p [get_pins -hierarchical -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin: $name"}
    return $p
}
proc v3_reference_alias {dcp output} {
    open_checkpoint $dcp
    set f [open $output {WRONLY CREAT EXCL}]
    puts $f "port\tdriver\tstartpoint"
    set previous {}
    foreach name {axi_aresetn axi_ctl_aresetn} {
        set p [v3_pin ntrpu_bridge_i/$name]
        set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $p] -filter {DIRECTION == OUT}]
        if {[llength $drivers]!=1 || ($previous ne {} && $drivers ne $previous)} {error "Reference resets are not aliases"}
        set previous $drivers
        puts $f "$name\t$drivers\t[join [all_fanin -flat -startpoints_only -to $p] ,]"
    }
    close $f
    set ::v3_reference_alias_verified 1
    close_project
}
proc v3_evidence {out} {
    file mkdir $out
    set f [open $out/native_reset_alias.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tdriver\tstartpoint"
    set previous {}
    foreach name {axi_aresetn axi_ctl_aresetn} {
        set p [get_pins -hierarchical -quiet -filter [format {NAME == "ntrpu_bridge_i/%s"} $name]]
        set nets {};set drivers {}
        if {[llength $p]} {set nets [get_nets -quiet -segments -of_objects $p]}
        if {[llength $nets]} {set drivers [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]}
        if {$name eq "axi_ctl_aresetn" && ![llength $drivers] && [info exists ::v3_reference_alias_verified]} {
            puts $f "$name\t\tREMOVED_UNUSED_ALIAS"
            continue
        }
        if {[llength $drivers]!=1 || ($previous ne {} && $drivers ne $previous)} {error "Native reset outputs are not aliases"}
        set previous $drivers
        puts $f "$name\t$drivers\t[join [all_fanin -flat -startpoints_only -to $p] ,]"
    }
    close $f
    set sync [get_cells -hierarchical -filter {NAME =~ guarded/guard/* && ASYNC_REG == 1}]
    if {[llength $sync]!=36} {error "Expected 36 ASYNC_REG registers"}
    set f [open $out/synchronizers.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tloc\tbel"
    foreach c [lsort $sync] {puts $f "$c\t[get_property REF_NAME $c]\t[get_property LOC $c]\t[get_property BEL $c]"}
    close $f
    set f [open $out/interstage.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tslack\tclock"
    set chains {cold_m cold_a echo_a echo_m armed_a stop_a}
    for {set i 0} {$i<3} {incr i} {lappend chains [format {availability[%d].release_m} $i] [format {availability[%d].release_a} $i]}
    foreach chain $chains {
        for {set i 0} {$i<2} {incr i} {
            set from [v3_pin [format {guarded/guard/%s_reg[%d]/C} $chain $i]]
            set to [v3_pin [format {guarded/guard/%s_reg[%d]/D} $chain [expr {$i+1}]]]
            foreach kind {max min} {
                set p [get_timing_paths -quiet -no_report_unconstrained -from $from -to $to -delay_type $kind -max_paths 1]
                if {[llength $p]!=1} {error "Missing V3 interstage timing"}
                puts $f "$from\t$to\t$kind\t[get_property SLACK $p]\t[get_property STARTPOINT_CLOCK $p]"
            }
        }
    }
    close $f
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_methodology -file $out/methodology.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/effective.xdc
}
