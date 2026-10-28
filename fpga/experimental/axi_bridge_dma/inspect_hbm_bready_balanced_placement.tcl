# Intermediate cluster experiment, using frozen shared and full-board audits.
source [file join [file dirname [info script]] inspect_hbm_bready_shared_placement.tcl]
proc lp_plan {} {
    set f [open [file join $::lp_source placement_plan.tsv] r]
    set lines [split [string trim [read $f]] \n];close $f
    if {[lindex $lines 0] ne "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel"} {error "Balanced placement header"}
    set rows {};foreach line [lrange $lines 1 end] {lappend rows [split $line \t]}
    set expected [list [list {weight_loader/bready_reg_reg[1]} {FDCE} {SLICE_X41Y11} {SLICEM.CFF} {SLICE_X100Y4} {CFF}] [list {weight_loader/bready_predictor/next_ready[1]_INST_0} {LUT4} {SLICE_X41Y11} {SLICEM.C6LUT} {SLICE_X100Y4} {C6LUT}] [list {weight_loader/bready_predictor/next_ready[1]_INST_0_i_1} {LUT5} {SLICE_X41Y11} {SLICEM.H6LUT} {SLICE_X100Y4} {H6LUT}] [list {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf_i_33} {LUT2} {SLICE_X113Y11} {SLICEL.C6LUT} {SLICE_X122Y0} {A6LUT}] [list {weight_loader/bready_predictor_i_2} {LUT2} {SLICE_X7Y11} {SLICEL.D5LUT} {SLICE_X63Y11} {F6LUT}]]
    if {$rows ne $expected} {error "Balanced placement plan differs"}
    return $rows
}
proc lp_stage {out} {
    if {$::lp_mode ni {eco reopen}} {error "Balanced placement supports eco/reopen only"}
    set rows [lp_plan]
    sp_inventory $out/placement_before
    lp_cells $out/cells_before.tsv
    lp_clock_pins $out/clock_pins_before.tsv
    sp_clock_route $out/clock_pips_before.tsv
    set occupancy [open $out/destination_before.tsv {WRONLY CREAT EXCL}];puts $occupancy "bel\tcells"
    set targets {}
    foreach row $rows {
        lassign $row name primitive oldloc oldbel loc bel
        set cell [cp_one [get_cells $name] balanced_cell]
        if {[get_property REF_NAME $cell] ne $primitive} {error "Donor primitive differs"}
        if {$::lp_mode eq "eco" && ([get_property LOC $cell] ne $oldloc || [get_property BEL $cell] ne $oldbel || [get_property IS_LOC_FIXED $cell] || [get_property IS_BEL_FIXED $cell])} {error "Balanced donor differs: $name"}
        lappend targets $loc/$bel
        if {[string match *6LUT $bel]} {lappend targets $loc/[string replace $bel 1 1 5]}
    }
    foreach target [lsort -unique $targets] {
        set obj [cp_one [get_bels $target] balanced_target]
        set occupants [get_cells -quiet -of_objects $obj]
        puts $occupancy "$target\t[join [lsort $occupants] ,]"
        if {$::lp_mode eq "eco" && [llength $occupants]} {error "Balanced destination occupied: $target"}
    }
    close $occupancy
    set f [open $out/placement_actions.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel"
    foreach row $rows {puts $f [join $row \t]};close $f
    if {$::lp_mode eq "eco"} {
        foreach row $rows {
            lassign $row name primitive oldloc oldbel loc bel
            place_cell [list $name $loc/$bel]
        }
        report_route_status -file $out/after_moves_route_status.rpt
        lp_repair_clock $out
        report_route_status -file $out/after_clock_repair_route_status.rpt
        write_checkpoint $out/placed.dcp
        route_design -preserve -no_psir
        route_design -finalize
        write_checkpoint $out/routed.dcp
    }
    foreach row $rows {
        lassign $row name primitive oldloc oldbel loc bel
        set cell [get_cells $name]
        if {[get_property LOC $cell] ne $loc || [lindex [split [get_property BEL $cell] .] end] ne $bel} {error "Balanced final placement differs: $name"}
    }
    sp_inventory $out/placement_after
    lp_cells $out/cells_after.tsv
    lp_clock_pins $out/clock_pins_after.tsv
    sp_clock_route $out/clock_pips_after.tsv
}
