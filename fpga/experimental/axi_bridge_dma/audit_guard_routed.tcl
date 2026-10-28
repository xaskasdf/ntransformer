# Read-only review of placer-inserted reset buffers and synchronizer timing.
set_param general.maxThreads 2
proc exact {name} {
    set p [get_pins -hierarchical -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin: $name"}
    return $p
}
if {[catch {
    if {[llength $argv]!=2} {error "ROUTED_CHECKPOINT NEW_OUTPUT required"}
    lassign $argv dcp out
    if {[file exists $out]} {error "New output required"};file mkdir $out
    open_checkpoint $dcp
    write_xdc -constraints all $out/before.xdc
    set f [open $out/buffers.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tloc\tbel\tinput_startpoints\tinput_drivers\tce_startpoints\tclock_loads\tdata_loads\treset_loads"
    foreach c [lsort [get_cells -hierarchical -filter {REF_NAME =~ BUFG*}]] {
        set inputs [get_pins -quiet -of_objects $c -filter {REF_PIN_NAME == I}]
        set ce [get_pins -quiet -of_objects $c -filter {REF_PIN_NAME == CE}]
        set start {};set drivers {};set enable {};set nc 0;set nd 0;set nr 0
        if {[llength $inputs]} {
            set start [all_fanin -flat -startpoints_only -to $inputs]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $inputs] -filter {DIRECTION == OUT}]
        }
        if {[llength $ce]} {set enable [all_fanin -flat -startpoints_only -to $ce]}
        set output [get_pins -of_objects $c -filter {DIRECTION == OUT}]
        foreach p [all_fanout -flat -endpoints_only -from $output] {
            if {[get_property CLASS $p] ne "pin"} {continue}
            if {[get_property IS_CLOCK $p]} {incr nc} elseif {[get_property REF_PIN_NAME $p] in {R S CLR PRE}} {incr nr} else {incr nd}
        }
        puts $f [join [list $c [get_property REF_NAME $c] [get_property LOC $c] [get_property BEL $c] [join $start ,] [join $drivers ,] [join $enable ,] $nc $nd $nr] "\t"]
        if {[string match *guarded/guard/cold_a_reg* $start]} {
            report_property -all -file $out/reset_buffer.rpt $c
            set p [get_pins -quiet -of_objects $c -filter {REF_PIN_NAME == CE}]
            set nets [get_nets -segments -of_objects $p]
            set d [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
            set fd [open $out/reset_enable.tsv {WRONLY CREAT EXCL}]
            puts $fd "buffer\tdriver\tdriver_ref\ti_inverted\tce_inverted\tce_type"
            foreach v $d {puts $fd "$c\t$v\t[get_property REF_NAME [get_cells -of_objects $v]]\t[get_property IS_I_INVERTED $c]\t[get_property IS_CE_INVERTED $c]\t[get_property CE_TYPE $c]"}
            close $fd
            if {$nc!=0} {error "Cold-reset buffer drives clock pins"}
        }
    }
    close $f
    set f [open $out/interstage.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tslack\tclock"
    set chains {cold_m cold_a echo_a echo_m armed_a stop_a}
    for {set n 0} {$n<4} {incr n} {lappend chains [format {availability[%d].release_m} $n] [format {availability[%d].release_a} $n]}
    foreach chain $chains {
        for {set i 0} {$i<2} {incr i} {
            set from [exact [format {guarded/guard/%s_reg[%d]/C} $chain $i]]
            set to [exact [format {guarded/guard/%s_reg[%d]/D} $chain [expr {$i+1}]]]
            foreach kind {max min} {
                set p [get_timing_paths -quiet -no_report_unconstrained -from $from -to $to -delay_type $kind -max_paths 1]
                if {[llength $p]!=1} {error "Missing interstage timing"}
                puts $f [join [list $from $to $kind [get_property SLACK $p] [get_property STARTPOINT_CLOCK $p]] "\t"]
            }
        }
    }
    close $f
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_clocks -file $out/clocks.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_route_status -file $out/route_status.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f ROUTED_GUARD_READONLY_AUDIT_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
