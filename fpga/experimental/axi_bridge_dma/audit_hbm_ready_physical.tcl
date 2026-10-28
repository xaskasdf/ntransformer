# Prove ordinary registered RREADY on both active HBM channels.
# Uses cp_one from the unchanged response capture auditor.
proc rp_driver {pin} {
    return [cp_one [get_pins -leaf -of_objects [get_nets -of_objects [cp_one [get_pins $pin] $pin]] -filter {DIRECTION == OUT}] driver]
}
proc rp_timing {out label from to type} {
    set query [list -to [get_pins $to] -delay_type $type -max_paths 1 -nworst 1]
    if {$from ne ""} {lappend query -from [get_pins $from]}
    set paths [get_timing_paths {*}$query]
    cp_one $paths $label
    ht_table $out $label $paths
}
proc rp_probe {out timed} {
    file mkdir $out
    set f [open $out/registers.tsv {WRONLY CREAT EXCL}]
    puts $f "channel\tobject\tproperty\tvalue"
    set owners [open $out/ownership.tsv {WRONLY CREAT EXCL}]
    puts $owners "channel\tcell\tpin\trvalid_dependency\tstartpoints"
    set dist core/reader/startup/memory_distribution_reg
    for {set ch 0} {$ch<2} {incr ch} {
        set tracker [format {core/reader/channels[%d].adapter/tracker} $ch]
        set name $tracker/rready_reg_reg
        set cell [cp_one [get_cells $name] ready_register]
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0"} {error "READY primitive changed"}
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED LOC BEL} {
            puts $f "$ch\t$name\t$key\t[get_property $key $cell]"
        }
        if {[get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {error "READY register must be ordinary"}
        set clock [get_clocks -of_objects [get_pins $name/C]]
        if {$clock ne "clk_out2_fabric_clocks"} {error "READY clock changed"}
        puts $f "$ch\t$name/C\tCLOCK\t$clock"
        set ce [rp_driver $name/CE]
        if {[get_property REF_NAME [get_cells -of_objects [get_pins $ce]]] ne "VCC"} {error "READY enable is not constant"}
        puts $f "$ch\t$name/CE\tDRIVER\t$ce"
        puts $f "$ch\t$name/CE\tDRIVER_PRIMITIVE\tVCC"
        set ready $tracker/h_rready
        if {[rp_driver $ready] ne "$name/Q"} {error "Combinational gate after READY"}
        puts $f "$ch\t$ready\tDRIVER\t$name/Q"
        set axi [format {hbm_binding/memory/AXI_%02d_RREADY} [expr {$ch*16}]]
        if {[rp_driver $axi] ne "$name/Q"} {error "External AXI READY binding changed"}
        puts $f "$ch\t$axi\tDRIVER\t$name/Q"
        set endpoint [format {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<%d>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST/RREADY} $ch]
        cp_one [get_pins $endpoint] hbm_ready_endpoint
        # AMD's unchanged hbm_top.sv drives primitive READY = AXI_READY || !AXI_VALID.
        # Prove that exact two-input function and both drivers, including pin order.
        set primitive [file dirname $endpoint]
        set valid_source [rp_driver $tracker/h_rvalid]
        if {$valid_source ne "$primitive/RVALID_PIPE"} {error "Unexpected HBM RVALID source"}
        set gate_pin [rp_driver $endpoint]
        set gate [cp_one [get_cells -of_objects [get_pins $gate_pin]] native_ready_gate]
        set init [get_property INIT $gate]
        if {[get_property REF_NAME $gate] ne "LUT2" || ![regexp {^4'h([0-9a-fA-F])$} $init unused hex]} {error "Unproven native READY gate"}
        set i0 [rp_driver $gate/I0];set i1 [rp_driver $gate/I1]
        scan $hex %x truth
        if {!(($i0 eq "$name/Q" && $i1 eq $valid_source && $truth==11) ||
              ($i1 eq "$name/Q" && $i0 eq $valid_source && $truth==13))} {error "Native READY function or inputs changed"}
        puts $f "$ch\t$endpoint\tDRIVER\t$gate_pin"
        puts $f "$ch\t$gate\tREF_NAME\tLUT2"
        puts $f "$ch\t$gate\tINIT\t$init"
        puts $f "$ch\t$gate/I0\tDRIVER\t$i0"
        puts $f "$ch\t$gate/I1\tDRIVER\t$i1"
        set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $endpoint]]]
        if {$starts ne [lsort [list $name/C $primitive/ACLK]]} {error "Native READY has additional fan-in"}
        puts $f "$ch\t$endpoint\tSTARTPOINTS\t[join $starts ,]"
        set clr [rp_driver $name/CLR]
        set inverted [get_property IS_INVERTED [get_pins $name/CLR]]
        puts $f "$ch\t$name/CLR\tDRIVER\t$clr"
        puts $f "$ch\t$name/CLR\tIS_INVERTED\t$inverted"
        set reset_starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/CLR]]]
        if {$reset_starts ne "$dist/C"} {error "READY reset bypasses distribution"}
        puts $f "$ch\t$name/CLR\tSTARTPOINTS\t$reset_starts"
        if {$clr eq "$dist/Q"} {
            if {$inverted ne "1"} {error "READY clear polarity"}
            puts $f "$ch\t$name\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
        } else {
            set inv [cp_one [get_cells -of_objects [get_pins $clr]] ready_reset_inverter]
            if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" ||
                $inverted ne "0" || [rp_driver $inv/I0] ne "$dist/Q"} {error "Unproven READY reset inverter"}
            puts $f "$ch\t$name\tRESET_ENCODING\tEXPLICIT_INVERTER"
            puts $f "$ch\t$inv\tREF_NAME\tLUT1"
            puts $f "$ch\t$inv\tINIT\t2'h1"
            puts $f "$ch\t$inv/I0\tDRIVER\t$dist/Q"
        }
        set valid_ends [all_fanout -flat -endpoints_only -from [get_pins $tracker/h_rvalid]]
        puts $f "$ch\t$tracker/h_rvalid\tDRIVER\t$valid_source"
        foreach tail {{count_reg[0]} {count_reg[1]} write_slot_reg} {
            set owner $tracker/response_cut/$tail
            cp_one [get_cells $owner] owner
            set depends 0
            foreach pin {CE D R} {
                set fanin [lsort [all_fanin -flat -startpoints_only -to [get_pins $owner/$pin]]]
                set valid_dep [expr {"$owner/$pin" in $valid_ends}]
                puts $owners "$ch\t$owner\t$pin\t$valid_dep\t[join $fanin ,]"
                if {$valid_dep} {set depends 1}
            }
            if {!$depends} {error "Ownership register lost HBM RVALID"}
        }
        if {$timed} {
            foreach type {max min} {
                rp_timing $out ch${ch}_input_$type "" $name/D $type
                rp_timing $out ch${ch}_output_$type $name/C $endpoint $type
                rp_timing $out ch${ch}_reset_$type $dist/C $name/CLR $type
            }
        }
    }
    close $f;close $owners
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f READY_BOTH_CHANNELS_EXPORTED;close $f
}
