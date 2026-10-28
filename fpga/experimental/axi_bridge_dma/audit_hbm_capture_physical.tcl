# Check both active HBM response queues in the complete native composition.
proc cp_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc cp_probe {out timed} {
    file mkdir $out
    set f [open $out/enables.tsv {WRONLY CREAT EXCL}]
    puts $f "channel\tcell\tprimitive\tclock\tdriver\tstartpoints"
    set vf [open $out/rvalid_endpoints.tsv {WRONLY CREAT EXCL}]
    puts $vf "channel\tendpoint"
    set cached [dict create]
    set all_ce {}
    for {set channel 0} {$channel<2} {incr channel} {
        set tracker [format {core/reader/channels[%d].adapter/tracker} $channel]
        set prefix $tracker/response_cut
        set valid [cp_one [get_pins $tracker/h_rvalid] h_rvalid]
        set owners 0
        foreach to [all_fanout -flat -endpoints_only -from $valid] {
            puts $vf "$channel\t$to"
            if {[string first "$prefix/slot" $to]==0 && [string match */CE $to]} {
                error "HBM RVALID still reaches wide response enable: $to"
            }
            if {[string first "$prefix/count_reg" $to]==0 ||
                [string first "$prefix/write_slot_reg" $to]==0} {incr owners}
        }
        if {$owners==0} {error "RVALID lost response ownership dependency"}
        for {set slot 0} {$slot<2} {incr slot} {
            for {set bit 0} {$bit<265} {incr bit} {
                set name [format {%s/slot%d_reg[%d]} $prefix $slot $bit]
                set cell [cp_one [get_cells $name] payload_register]
                if {[get_property REF_NAME $cell] ne "FDRE"} {error "Payload primitive changed"}
                set ce [get_pins -of_objects $cell -filter {REF_PIN_NAME == CE}]
                cp_one $ce payload_enable
                set net [get_nets -of_objects $ce]
                set key [get_property NAME $net]
                if {![dict exists $cached $key]} {
                    set starts [lsort [all_fanin -flat -startpoints_only -to $ce]]
                    set driver [cp_one [get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}] enable_driver]
                    dict set cached $key [list $driver $starts]
                }
                lassign [dict get $cached $key] driver starts
                if {![llength $starts]} {error "Empty payload enable cone"}
                foreach start $starts {
                    if {$start ne "core/reader/startup/memory_distribution_reg/C" &&
                        [string first "$prefix/count_reg" $start]!=0 &&
                        $start ne "$prefix/write_slot_reg/C"} {error "Unexpected free-slot enable input: $start"}
                }
                set clock [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
                if {$clock ne "clk_out2_fabric_clocks"} {error "Payload clock differs"}
                puts $f "$channel\t$name\tFDRE\t$clock\t$driver\t[join $starts ,]"
                lappend all_ce $ce
            }
        }
    }
    close $f;close $vf
    if {$timed} {
        foreach type {max min} {
            set paths [get_timing_paths -to $all_ce -delay_type $type -max_paths 1060 -nworst 1]
            if {[llength $paths]!=1060} {error "Incomplete response timing sample"}
            set f [open $out/enable_${type}.tsv {WRONLY CREAT EXCL}]
            puts $f "from\tto\tclock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
            foreach path $paths {
                puts $f "[get_property STARTPOINT_PIN $path]\t[get_property ENDPOINT_PIN $path]\t[get_property STARTPOINT_CLOCK $path]\t[get_property ENDPOINT_CLOCK $path]\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]\t[get_property DATAPATH_DELAY $path]"
            };close $f
        }
        report_timing -to $all_ce -delay_type min_max -max_paths 8 -file $out/enable_timing.rpt
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f CAPTURE_BOTH_CHANNELS_EXPORTED;close $f
}
