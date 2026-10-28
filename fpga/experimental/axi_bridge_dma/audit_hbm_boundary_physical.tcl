# Read-only audit. Inherits cp_one/rp_driver/rp_timing/pp_dcone/ht_table.
proc bp_probe {out timed candidate} {
    file mkdir $out
    set f [open $out/registers.tsv {WRONLY CREAT EXCL}];puts $f "channel\tobject\tproperty\tvalue"
    set pf [open $out/payload.tsv {WRONLY CREAT EXCL}];puts $pf "channel\tcell\tprimitive\tclock\tstartpoints"
    set pr [open $out/pruned.tsv {WRONLY CREAT EXCL}];puts $pr "channel\tslot\tbit\tport\tnets\tdriver\tprimitive"
    set vf [open $out/valid_endpoints.tsv {WRONLY CREAT EXCL}];puts $vf "channel\tendpoint"
    set of [open $out/ownership.tsv {WRONLY CREAT EXCL}];puts $of "channel\tcell\tpin\tvalid_dependency"
    set dist core/reader/startup/memory_distribution_reg
    set cached [dict create]
    for {set ch 0} {$ch<2} {incr ch} {
        set adapter [format {core/reader/channels[%d].adapter} $ch]
        set tracker $adapter/tracker;set queue $tracker/validated_cut
        set ready $tracker/rready_reg_reg;set valid $queue/m_valid
        pp_dcone $f $ch $ready
        set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $ready/D]]]
        # Hierarchical channel names contain literal brackets, not glob classes.
        set lengths {}
        foreach start $starts {
            if {[string first "$tracker/h_arlen_reg" $start]==0} {lappend lengths $start}
        }
        puts $f "$ch\t$ready/D\tARLEN_STARTPOINTS\t[join $lengths ,]"
        if {$candidate && [llength $lengths]} {error "READY credit adder remains"}
        if {!$candidate && ![llength $lengths]} {error "Reference credit adder absent"}
        set driver [rp_driver $valid];set cell [cp_one [get_cells -of_objects [get_pins $driver]] valid_cell]
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH LOC BEL} {puts $f "$ch\t$cell\t$key\t[get_property $key $cell]"}
        foreach pin [list $valid $tracker/p_rvalid $adapter/protocol_converter/m_axi_rvalid] {
            if {[rp_driver $pin] ne $driver} {error "Validated VALID interface binding differs: $pin"}
            puts $f "$ch\t$pin\tDRIVER\t$driver"
        }
        if {$candidate} {
            set name $queue/valid_reg_reg
            if {$driver ne "$name/Q" || [get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0" ||
                [get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {error "Validated control register differs"}
            set clock [get_clocks -of_objects [get_pins $name/C]]
            set ce [get_property REF_NAME [get_cells -of_objects [get_pins [rp_driver $name/CE]]]]
            if {$clock ne "clk_out2_fabric_clocks" || $ce ne "VCC"} {error "Validated clock or enable differs"}
            puts $f "$ch\t$name/C\tCLOCK\t$clock"
            puts $f "$ch\t$name/CE\tDRIVER_PRIMITIVE\t$ce"
            pp_dcone $f $ch $name
            set clr [rp_driver $name/CLR];set inverted [get_property IS_INVERTED [get_pins $name/CLR]]
            set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/CLR]]]
            if {$starts ne "$dist/C"} {error "Validated reset bypasses distribution"}
            puts $f "$ch\t$name/CLR\tSTARTPOINTS\t$starts"
            puts $f "$ch\t$name/CLR\tDRIVER\t$clr"
            puts $f "$ch\t$name/CLR\tIS_INVERTED\t$inverted"
            if {$clr eq "$dist/Q"} {
                if {$inverted ne "1"} {error "Validated reset polarity"}
                puts $f "$ch\t$name\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
            } else {
                set inv [cp_one [get_cells -of_objects [get_pins $clr]] reset_inverter]
                if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" ||
                    $inverted ne "0" || [rp_driver $inv/I0] ne "$dist/Q"} {error "Unproven validated reset inverter"}
                puts $f "$ch\t$name\tRESET_ENCODING\tEXPLICIT_INVERTER"
                puts $f "$ch\t$inv\tREF_NAME\tLUT1"
                puts $f "$ch\t$inv\tINIT\t2'h1"
                puts $f "$ch\t$inv/I0\tDRIVER\t$dist/Q"
            }
        } elseif {![string match LUT* [get_property REF_NAME $cell]]} {error "Baseline validated VALID is no longer combinational"}
        set ends [lsort [all_fanout -flat -endpoints_only -from [get_pins $queue/s_valid]]]
        foreach pin $ends {puts $vf "$ch\t$pin"}
        foreach tail {{count_reg[0]} {count_reg[1]} write_slot_reg} {
            set owner $queue/$tail;cp_one [get_cells $owner] owner;set observed 0
            foreach pin {CE D R} {
                set depends [expr {"$owner/$pin" in $ends}]
                puts $of "$ch\t$owner\t$pin\t$depends"
                if {$depends} {set observed 1}
            }
            if {!$observed} {error "Validated ownership lost input VALID"}
        }
        for {set slot 0} {$slot<2} {incr slot} {
            for {set bit 0} {$bit<265} {incr bit} {
                set name [format {%s/slot%d_reg[%d]} $queue $slot $bit]
                set cells [get_cells -quiet $name]
                if {![llength $cells]} {
                    if {$bit<259} {error "Validated data/last/response bit absent"}
                    set pins [list [format {%s/m_data[%d]} $queue $bit] [format {%s/validated_rid[%d]} $tracker [expr {$bit-259}]]]
                    if {$bit<263} {lappend pins [format {%s/protocol_converter/m_axi_rid[%d]} $adapter [expr {$bit-259}]]}
                    foreach pin $pins {
                        set obj [cp_one [get_pins $pin] pruned_id_pin]
                        set nets [get_nets -quiet -of_objects $obj];set drv "";set type UNCONNECTED
                        if {[llength $nets]} {
                            set drv [rp_driver $pin];set type [get_property REF_NAME [get_cells -of_objects [get_pins $drv]]]
                            if {$type ni {GND VCC}} {error "Pruned validated ID remains dynamic: $pin"}
                        }
                        puts $pr "$ch\t$slot\t$bit\t$pin\t[llength $nets]\t$drv\t$type"
                    }
                    continue
                }
                set c [cp_one $cells payload];set ce [get_pins $name/CE];set key [get_nets -of_objects $ce]
                if {![dict exists $cached $key]} {dict set cached $key [lsort [all_fanin -flat -startpoints_only -to $ce]]}
                set clock [get_clocks -of_objects [get_pins $name/C]]
                if {[get_property REF_NAME $c] ne "FDRE" || $clock ne "clk_out2_fabric_clocks"} {error "Validated payload primitive or clock differs"}
                puts $pf "$ch\t$name\tFDRE\t$clock\t[join [dict get $cached $key] ,]"
            }
        }
        if {$timed} {
            foreach type {max min} {
                set target $adapter/return_cut/valid_reg_reg/D
                set path [get_timing_paths -through [get_pins $valid] -to [get_pins $target] -delay_type $type -max_paths 1 -nworst 1]
                cp_one $path boundary_timing;ht_table $out ch${ch}_output_$type $path
                if {$candidate} {
                    rp_timing $out ch${ch}_input_$type "" $queue/valid_reg_reg/D $type
                    rp_timing $out ch${ch}_reset_$type $dist/C $queue/valid_reg_reg/CLR $type
                }
            }
        }
    }
    foreach handle [list $f $pf $pr $vf $of] {close $handle}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f [expr {$candidate?"BOUNDARY_CANDIDATE_EXPORTED":"BOUNDARY_BASELINE_EXPORTED"}];close $f
}
