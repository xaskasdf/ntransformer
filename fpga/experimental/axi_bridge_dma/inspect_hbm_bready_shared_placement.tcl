# Extend frozen board audits; this experiment moves one combinational LUT.
source [file join [file dirname [info script]] inspect_hbm_bready_placement.tcl]

proc lp_plan {} {
    set f [open [file join $::lp_source placement_plan.tsv] r]
    set lines [split [string trim [read $f]] \n];close $f
    if {[lindex $lines 0] ne "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel" || [llength $lines]!=2} {error "Single shared-LUT plan required"}
    set row [split [lindex $lines 1] \t]
    if {[llength $row]!=6 || [lrange $row 0 3] ne {weight_loader/bready_predictor_i_2 LUT2 SLICE_X7Y11 SLICEL.D5LUT}} {error "Unexpected shared predictor donor"}
    if {[lrange $row 4 5] ne {SLICE_X63Y11 F6LUT}} {error "Unexpected shared predictor destination"}
    return [list $row]
}

proc sp_inventory {out} {
    file mkdir $out
    set cone {}
    foreach ch {0 1} {
        set reg [format {weight_loader/bready_reg_reg[%d]} $ch]
        set native [format {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<%d>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST} $ch]
        set gate [get_cells -of_objects [get_pins [rp_driver $native/BREADY]]]
        set cone [concat $cone [all_fanin -flat -only_cells -to [get_pins $reg/D]] [list $reg $gate $native]]
    }
    set f [open $out/cone.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tloc\tbel\tinit\tfixed_loc\tfixed_bel"
    set links [open $out/connections.tsv {WRONLY CREAT EXCL}]
    puts $links "cell\tpin\tdirection\tnets"
    foreach name [lsort -unique $cone] {
        set cell [cp_one [get_cells $name] shared_cone_cell]
        puts $f [join [list $cell [get_property REF_NAME $cell] [get_property LOC $cell] [get_property BEL $cell] [get_property INIT $cell] [get_property IS_LOC_FIXED $cell] [get_property IS_BEL_FIXED $cell]] \t]
        foreach pin_name [lsort [get_pins -of_objects $cell]] {
            set pin [get_pins $pin_name]
            puts $links [join [list $cell $pin [get_property DIRECTION $pin] [join [lsort [get_nets -segments -of_objects $pin]] ,]] \t]
        }
    }
    close $f;close $links
    set f [open $out/shared_endpoints.tsv {WRONLY CREAT EXCL}];puts $f "pin"
    foreach pin [lsort [all_fanout -flat -endpoints_only -from [get_pins weight_loader/bready_predictor_i_2/O]]] {puts $f $pin}
    close $f
    set f [open $out/shared_drivers.tsv {WRONLY CREAT EXCL}];puts $f "pin\tdriver"
    foreach pin {I0 I1} {puts $f "weight_loader/bready_predictor_i_2/$pin\t[rp_driver weight_loader/bready_predictor_i_2/$pin]"}
    close $f
}

proc sp_clock_route {path} {
    set net [cp_one [get_nets {clock_generator/inst/clk_out2}] shared_clock]
    set f [open $path {WRONLY CREAT EXCL}];puts $f "pip"
    foreach pip [lsort [get_pips -of_objects $net]] {puts $f $pip}
    close $f
}

proc lp_stage {out} {
    if {$::lp_mode ni {eco reopen}} {error "Shared placement supports eco/reopen only"}
    set row [lindex [lp_plan] 0]
    lassign $row name primitive oldloc oldbel loc bel
    sp_inventory $out/placement_before
    lp_cells $out/cells_before.tsv
    lp_clock_pins $out/clock_pins_before.tsv
    sp_clock_route $out/clock_pips_before.tsv
    set cell [cp_one [get_cells $name] shared_predictor]
    if {[get_property REF_NAME $cell] ne "LUT2" || [get_property INIT $cell] ne "4'h1"} {error "Shared predictor function changed"}
    set f [open $out/destination_before.tsv {WRONLY CREAT EXCL}]
    puts $f "bel\tcells"
    foreach candidate [list $loc/F5LUT $loc/F6LUT] {
        set target [cp_one [get_bels $candidate] shared_target]
        set occupants [get_cells -quiet -of_objects $target]
        puts $f "$candidate\t[join [lsort $occupants] ,]"
        if {$::lp_mode eq "eco" && [llength $occupants]} {error "Shared LUT destination occupied"}
    }
    close $f
    set f [open $out/placement_actions.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel"
    puts $f [join $row \t];close $f
    if {$::lp_mode eq "eco"} {
        if {[get_property LOC $cell] ne $oldloc || [get_property BEL $cell] ne $oldbel || [get_property IS_LOC_FIXED $cell] || [get_property IS_BEL_FIXED $cell]} {error "Shared LUT donor placement changed"}
        place_cell [list $name $loc/$bel]
        report_route_status -file $out/after_move_route_status.rpt
        route_design -preserve -no_psir
        route_design -finalize
        write_checkpoint $out/routed.dcp
    }
    if {[get_property LOC $cell] ne $loc || [lindex [split [get_property BEL $cell] .] end] ne $bel} {error "Shared LUT final placement differs"}
    sp_inventory $out/placement_after
    lp_cells $out/cells_after.tsv
    lp_clock_pins $out/clock_pins_after.tsv
    sp_clock_route $out/clock_pips_after.tsv
}
