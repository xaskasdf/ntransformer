# Read-only full-board predictor, return ownership and CDC write-enable evidence.
# Requires cp_one, rp_driver, rp_timing and (for timed stages) ht_table.
proc pp_dcone {f ch cell} {
    set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $cell/D]]]
    if {![llength $starts]} {error "Empty control D cone"}
    puts $f "$ch\t$cell/D\tSTARTPOINTS\t[join $starts ,]"
    # External IP can reintroduce reset through its outputs. Report complete
    # board fan-in, rather than pretending the unit result proves its absence.
    puts $f "$ch\t$cell/D\tDISTRIBUTION_DEPENDENCY\t[expr {"core/reader/startup/memory_distribution_reg/C" in $starts}]"
}
proc pp_probe {out timed candidate} {
    file mkdir $out
    set f [open $out/registers.tsv {WRONLY CREAT EXCL}];puts $f "channel\tobject\tproperty\tvalue"
    set pf [open $out/payload.tsv {WRONLY CREAT EXCL}];puts $pf "channel\tcell\tprimitive\tclock\tstartpoints"
    set pruned [open $out/pruned.tsv {WRONLY CREAT EXCL}];puts $pruned "channel\tslot\tbit\tport\tdriver\tprimitive"
    set interfaces [open $out/pruned_interfaces.tsv {WRONLY CREAT EXCL}];puts $interfaces "channel\tslot\tbit\tport\tnets\tdriver\tprimitive"
    set ef [open $out/endpoints.tsv {WRONLY CREAT EXCL}];puts $ef "channel\tendpoint\tprimitive\tpin\twrite_enable"
    set of [open $out/ownership.tsv {WRONLY CREAT EXCL}];puts $of "channel\tcell\tpin\tvalid_dependency"
    set vf [open $out/input_valid_endpoints.tsv {WRONLY CREAT EXCL}];puts $vf "channel\tendpoint"
    set dist core/reader/startup/memory_distribution_reg
    set cached [dict create]
    for {set ch 0} {$ch<2} {incr ch} {
        set adapter [format {core/reader/channels[%d].adapter} $ch]
        set crossing [format {core/reader/channels[%d].crossing} $ch]
        set queue $adapter/return_cut;set valid $queue/m_valid
        pp_dcone $f $ch $adapter/tracker/rready_reg_reg
        set driver [rp_driver $valid];set cell [cp_one [get_cells -of_objects [get_pins $driver]] valid_cell]
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH LOC BEL} {puts $f "$ch\t$cell\t$key\t[get_property $key $cell]"}
        foreach pin [list $valid $adapter/s_rvalid $crossing/m_axi_rvalid] {
            if {[rp_driver $pin] ne $driver} {error "Return VALID binding differs"}
            puts $f "$ch\t$pin\tDRIVER\t$driver"
        }
        if {$candidate} {
            set name $queue/valid_reg_reg
            if {$cell ne $name || $driver ne "$name/Q" || [get_property REF_NAME $cell] ne "FDCE" ||
                [get_property INIT $cell] ne "1'b0"} {error "Return VALID is not the tested register"}
            if {[get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {error "Return VALID must be ordinary"}
            set clock [get_clocks -of_objects [get_pins $name/C]]
            if {$clock ne "clk_out2_fabric_clocks"} {error "Return VALID clock differs"}
            puts $f "$ch\t$name/C\tCLOCK\t$clock"
            set ce [get_cells -of_objects [get_pins [rp_driver $name/CE]]]
            if {[get_property REF_NAME $ce] ne "VCC"} {error "Return VALID enable differs"}
            puts $f "$ch\t$name/CE\tDRIVER_PRIMITIVE\tVCC"
            pp_dcone $f $ch $name
            set clr [rp_driver $name/CLR];set inverted [get_property IS_INVERTED [get_pins $name/CLR]]
            set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/CLR]]]
            if {$starts ne "$dist/C"} {error "Return reset bypasses distribution"}
            puts $f "$ch\t$name/CLR\tSTARTPOINTS\t$starts"
            puts $f "$ch\t$name/CLR\tDRIVER\t$clr"
            puts $f "$ch\t$name/CLR\tIS_INVERTED\t$inverted"
            if {$clr eq "$dist/Q"} {
                if {$inverted ne "1"} {error "Return reset polarity"}
                puts $f "$ch\t$name\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
            } else {
                set inv [cp_one [get_cells -of_objects [get_pins $clr]] return_reset_inverter]
                if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" ||
                    $inverted ne "0" || [rp_driver $inv/I0] ne "$dist/Q"} {error "Unproven return reset inverter"}
                puts $f "$ch\t$name\tRESET_ENCODING\tEXPLICIT_INVERTER"
                puts $f "$ch\t$inv\tREF_NAME\tLUT1"
                puts $f "$ch\t$inv\tINIT\t2'h1"
                puts $f "$ch\t$inv/I0\tDRIVER\t$dist/Q"
            }
        } elseif {![string match LUT* [get_property REF_NAME $cell]]} {error "Baseline VALID no longer combinational"}
        set valid_ends [all_fanout -flat -endpoints_only -from [get_pins $queue/s_valid]]
        foreach pin $valid_ends {puts $vf "$ch\t$pin"}
        foreach tail {{count_reg[0]} {count_reg[1]} write_slot_reg} {
            set owner $queue/$tail;cp_one [get_cells $owner] owner;set depends 0
            foreach pin {CE D R} {
                set dep [expr {"$owner/$pin" in $valid_ends}]
                puts $of "$ch\t$owner\t$pin\t$dep"
                if {$dep} {set depends 1}
            }
            if {!$depends} {error "Return ownership lost VALID"}
        }
        for {set slot 0} {$slot<2} {incr slot} {
            for {set bit 0} {$bit<519} {incr bit} {
                set name [format {%s/slot%d_reg[%d]} $queue $slot $bit]
                set cells [get_cells -quiet $name]
                if {![llength $cells]} {
                    if {$bit<515} {error "Return data/response/last bit is missing"}
                    set port [format {%s/m_data[%d]} $queue $bit]
                    # Only RID bits may disappear. Vivado can retain a logical
                    # pin while disconnecting it, or tie the parent interface
                    # directly to a constant. Prove each boundary separately.
                    foreach pin [list $port [format {%s/s_rid[%d]} $adapter [expr {$bit-515}]] [format {%s/m_axi_rid[%d]} $crossing [expr {$bit-515}]]] {
                        set obj [cp_one [get_pins $pin] pruned_id_pin]
                        set nets [get_nets -quiet -of_objects $obj];set drv "";set type UNCONNECTED
                        if {[llength $nets]} {
                            set drv [rp_driver $pin];set type [get_property REF_NAME [get_cells -of_objects [get_pins $drv]]]
                            if {$type ni {GND VCC}} {error "Pruned return ID remains dynamic: $pin"}
                        }
                        puts $interfaces "$ch\t$slot\t$bit\t$pin\t[llength $nets]\t$drv\t$type"
                        if {$pin eq $port} {puts $pruned "$ch\t$slot\t$bit\t$port\t$drv\t$type"}
                    }
                    continue
                }
                set c [cp_one $cells payload];set ce [get_pins $name/CE];set key [get_nets -of_objects $ce]
                if {![dict exists $cached $key]} {dict set cached $key [lsort [all_fanin -flat -startpoints_only -to $ce]]}
                set starts [dict get $cached $key]
                if {[get_property REF_NAME $c] ne "FDRE" || ![llength $starts]} {error "Return storage primitive differs"}
                foreach start $starts {
                    if {$start ni [list $dist/C $queue/count_reg\[0\]/C $queue/count_reg\[1\]/C $queue/write_slot_reg/C]} {error "Unexpected return payload enable input: $start"}
                }
                set clock [get_clocks -of_objects [get_pins $name/C]]
                if {$clock ne "clk_out2_fabric_clocks"} {error "Return payload clock differs"}
                puts $pf "$ch\t$name\tFDRE\t$clock\t[join $starts ,]"
            }
        }
        set we {}
        foreach pin [lsort [all_fanout -flat -endpoints_only -from [get_pins $valid]]] {
            set dst [get_pins $pin];set type [get_property REF_NAME [get_cells -of_objects $dst]]
            set refpin [get_property REF_PIN_NAME $dst]
            set is_we [expr {[string first "$crossing/" $pin]==0 && $type in {RAMD32 RAMS32} && $refpin eq "WE"}]
            puts $ef "$ch\t$pin\t$type\t$refpin\t$is_we"
            if {$is_we} {lappend we $dst}
        }
        if {![llength $we]} {error "CDC write-enable fanout is empty"}
        if {$timed} {
            foreach type {max min} {
                set paths [get_timing_paths -to $we -delay_type $type -max_paths [llength $we] -nworst 1]
                if {[llength $paths]!=[llength $we]} {error "Incomplete CDC WE timing coverage"}
                ht_table $out ch${ch}_cdc_we_$type $paths
                set through [get_timing_paths -through [get_pins $valid] -to $we -delay_type $type -max_paths 1 -nworst 1]
                cp_one $through valid_output_timing
                ht_table $out ch${ch}_return_output_$type $through
                if {$candidate} {
                    rp_timing $out ch${ch}_return_input_$type "" $queue/valid_reg_reg/D $type
                    rp_timing $out ch${ch}_return_reset_$type $dist/C $queue/valid_reg_reg/CLR $type
                }
            }
        }
    }
    foreach file [list $f $pf $pruned $interfaces $ef $of $vf] {close $file}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f [expr {$candidate?"CONTROLS_CANDIDATE_EXPORTED":"CONTROLS_BASELINE_EXPORTED"}];close $f
}
