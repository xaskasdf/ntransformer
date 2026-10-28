# Read-only audit of the tested BREADY output registers and native vendor gate.
# Inherits cp_one, rp_driver, rp_timing, pp_dcone and ht_table.
proc wb_probe {out timed} {
    file mkdir $out
    set f [open $out/registers.tsv {WRONLY CREAT EXCL}]
    puts $f "channel\tobject\tproperty\tvalue"
    set dist core/reader/startup/memory_distribution_reg
    for {set ch 0} {$ch<2} {incr ch} {
        set name [format {weight_loader/bready_reg_reg[%d]} $ch]
        set cell [cp_one [get_cells $name] bready_register]
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0"} {error "BREADY primitive changed"}
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED LOC BEL} {puts $f "$ch\t$name\t$key\t[get_property $key $cell]"}
        if {[get_property ASYNC_REG $cell] in {1 true TRUE} || [get_property DONT_TOUCH $cell] in {1 true TRUE}} {error "BREADY must be ordinary"}
        set clock [get_clocks -of_objects [get_pins $name/C]]
        if {$clock ne "clk_out2_fabric_clocks"} {error "BREADY clock changed"}
        puts $f "$ch\t$name/C\tCLOCK\t$clock"
        set ce [get_property REF_NAME [get_cells -of_objects [get_pins [rp_driver $name/CE]]]]
        if {$ce ne "VCC"} {error "BREADY enable changed"}
        puts $f "$ch\t$name/CE\tDRIVER_PRIMITIVE\t$ce"
        pp_dcone $f $ch $name
        set port [format {weight_loader/h_bready[%d]} $ch]
        set axi [format {hbm_binding/memory/AXI_%02d_BREADY} [expr {$ch*16}]]
        foreach pin [list $port $axi] {
            if {[rp_driver $pin] ne "$name/Q"} {error "BREADY output bypass: $pin"}
            puts $f "$ch\t$pin\tDRIVER\t$name/Q"
        }
        set primitive [format {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<%d>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST} $ch]
        set endpoint $primitive/BREADY
        set valid [format {weight_loader/h_bvalid[%d]} $ch]
        set valid_source [rp_driver $valid]
        if {$valid_source ne "$primitive/BVALID_PIPE"} {error "HBM BVALID source changed"}
        puts $f "$ch\t$valid\tDRIVER\t$valid_source"
        set gate_pin [rp_driver $endpoint]
        set gate [cp_one [get_cells -of_objects [get_pins $gate_pin]] native_bready_gate]
        set init [get_property INIT $gate]
        if {[get_property REF_NAME $gate] ne "LUT2" || ![regexp {^4'h([0-9a-fA-F])$} $init unused hex]} {error "Unproven native BREADY gate"}
        scan $hex %x truth
        set i0 [rp_driver $gate/I0];set i1 [rp_driver $gate/I1]
        if {!(($i0 eq "$name/Q" && $i1 eq $valid_source && $truth==11) || ($i1 eq "$name/Q" && $i0 eq $valid_source && $truth==13))} {error "Native BREADY function changed"}
        puts $f "$ch\t$endpoint\tDRIVER\t$gate_pin"
        puts $f "$ch\t$gate\tREF_NAME\tLUT2"
        puts $f "$ch\t$gate\tINIT\t$init"
        puts $f "$ch\t$gate/I0\tDRIVER\t$i0"
        puts $f "$ch\t$gate/I1\tDRIVER\t$i1"
        set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $endpoint]]]
        if {$starts ne [lsort [list $name/C $primitive/ACLK]]} {error "Native BREADY additional fan-in"}
        puts $f "$ch\t$endpoint\tSTARTPOINTS\t[join $starts ,]"
        set clr [rp_driver $name/CLR];set inverted [get_property IS_INVERTED [get_pins $name/CLR]]
        set starts [lsort [all_fanin -flat -startpoints_only -to [get_pins $name/CLR]]]
        if {$starts ne "$dist/C"} {error "BREADY cold reset bypasses distribution"}
        puts $f "$ch\t$name/CLR\tDRIVER\t$clr"
        puts $f "$ch\t$name/CLR\tIS_INVERTED\t$inverted"
        puts $f "$ch\t$name/CLR\tSTARTPOINTS\t$starts"
        if {$clr eq "$dist/Q"} {
            if {$inverted ne "1"} {error "BREADY clear polarity"}
            puts $f "$ch\t$name\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
        } else {
            set inv [cp_one [get_cells -of_objects [get_pins $clr]] bready_reset_inverter]
            if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" || $inverted ne "0" || [rp_driver $inv/I0] ne "$dist/Q"} {error "Unproven BREADY reset inverter"}
            puts $f "$ch\t$name\tRESET_ENCODING\tEXPLICIT_INVERTER"
            puts $f "$ch\t$inv\tREF_NAME\tLUT1"
            puts $f "$ch\t$inv\tINIT\t2'h1"
            puts $f "$ch\t$inv/I0\tDRIVER\t$dist/Q"
        }
        if {$timed} {
            foreach type {max min} {
                rp_timing $out ch${ch}_input_$type "" $name/D $type
                rp_timing $out ch${ch}_output_$type $name/C $endpoint $type
                rp_timing $out ch${ch}_native_$type "" $endpoint $type
                rp_timing $out ch${ch}_vendor_$type $primitive/ACLK $endpoint $type
                rp_timing $out ch${ch}_reset_$type $dist/C $name/CLR $type
            }
        }
    }
    close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f BREADY_BOTH_CHANNELS_EXPORTED;close $f
}
