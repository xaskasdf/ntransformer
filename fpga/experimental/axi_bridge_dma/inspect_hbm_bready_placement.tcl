# Isolated physical inspection; frozen BREADY and board audits remain intact.
source [file join [file dirname [info script]] implement_hbm_bready_physical.tcl]

proc lp_inventory {out} {
    file mkdir $out
    set reg {weight_loader/bready_reg_reg[1]}
    set native {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<1>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST}
    set gate [get_cells -of_objects [get_pins [rp_driver $native/BREADY]]]
    set cone [lsort -unique [concat [all_fanin -flat -only_cells -to [get_pins $reg/D]] [list $reg $gate $native]]]
    set f [open $out/cone.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tloc\tbel\tinit\tfixed_loc\tfixed_bel"
    set pins [open $out/connections.tsv {WRONLY CREAT EXCL}]
    puts $pins "cell\tpin\tdirection\tnets"
    foreach name $cone {
        set cell [cp_one [get_cells $name] cone_cell]
        puts $f [join [list $cell [get_property REF_NAME $cell] [get_property LOC $cell] [get_property BEL $cell] [get_property INIT $cell] [get_property IS_LOC_FIXED $cell] [get_property IS_BEL_FIXED $cell]] "\t"]
        foreach pin_name [lsort [get_pins -of_objects $cell]] {
            set pin [cp_one [get_pins $pin_name] cone_pin]
            puts $pins [join [list $cell $pin [get_property DIRECTION $pin] [join [lsort [get_nets -segments -of_objects $pin]] ,]] "\t"]
        }
    }
    close $f;close $pins
    set sites {}
    for {set x 90} {$x<=135} {incr x} {
        for {set y 0} {$y<=24} {incr y} {lappend sites SLICE_X${x}Y${y}}
    }
    set f [open $out/sites.tsv {WRONLY CREAT EXCL}]
    puts $f "site\ttype\ttile\tx\ty\tcells"
    foreach site_name [lsort [get_sites -quiet $sites]] {
        set site [cp_one [get_sites $site_name] placement_site]
        set tile [get_tiles -of_objects $site]
        set cells [lsort [get_cells -quiet -of_objects [get_bels -of_objects $site]]]
        puts $f [join [list $site [get_property SITE_TYPE $site] $tile [get_property GRID_POINT_X $tile] [get_property GRID_POINT_Y $tile] [join $cells ,]] "\t"]
    }
    close $f
    set f [open $out/native_site.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tsite\ttile\tx\ty"
    set site [get_sites -of_objects [get_cells $native]];set tile [get_tiles -of_objects $site]
    puts $f [join [list $native $site $tile [get_property GRID_POINT_X $tile] [get_property GRID_POINT_Y $tile]] "\t"]
    close $f
}

proc lp_cells {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tinit\tloc\tbel\tfixed_loc\tfixed_bel"
    set init_properties [dict create]
    foreach cell [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set primitive [get_property REF_NAME $cell]
        if {![dict exists $init_properties $primitive]} {
            dict set init_properties $primitive [expr {"INIT" in [list_property $cell]}]
        }
        set init ""
        if {[dict get $init_properties $primitive]} {set init [get_property INIT $cell]}
        puts $f [join [list $cell $primitive $init [get_property LOC $cell] [get_property BEL $cell] [get_property IS_LOC_FIXED $cell] [get_property IS_BEL_FIXED $cell]] "\t"]
    }
    close $f
}

proc lp_plan {} {
    set f [open [file join $::lp_source placement_plan.tsv] r]
    set text [split [string trim [read $f]] \n];close $f
    if {[lindex $text 0] ne "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel"} {error "Placement plan header"}
    set rows {};set names {};set destinations {}
    foreach line [lrange $text 1 end] {
        set row [split $line \t]
        if {[llength $row]!=6} {error "Malformed placement row"}
        lassign $row name primitive oldloc oldbel loc bel
        if {$name in $names || "$loc/$bel" in $destinations} {error "Duplicate placement target"}
        if {!(($name eq {weight_loader/bready_reg_reg[1]} && $primitive eq "FDCE") ||
              ([string match weight_loader/bready* $name] && [string match LUT? $primitive]) ||
              ($name eq "hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf_i_33" && $primitive eq "LUT2"))} {error "Out-of-scope placement cell: $name"}
        lappend rows $row;lappend names $name;lappend destinations "$loc/$bel"
    }
    if {[llength $rows]<2 || [llength $rows]>6} {error "Placement scope size"}
    return $rows
}

proc lp_clock_pins {path} {
    set net [cp_one [get_nets {clock_generator/inst/clk_out2}] affected_clock]
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "net\tpin\tdirection"
    foreach name [lsort [get_pins -leaf -of_objects $net]] {
        set pin [cp_one [get_pins $name] clock_pin]
        puts $f [join [list $net $pin [get_property DIRECTION $pin]] \t]
    }
    close $f
}

proc lp_repair_clock {out} {
    # Rebuild only the affected clock after removing its old routed tree.
    # UG835 requires this cleanup before updating a changed global clock.
    set net [cp_one [get_nets {clock_generator/inst/clk_out2}] affected_clock]
    set f [open $out/clock_repair.tsv {WRONLY CREAT EXCL}]
    puts $f "net\tcommand"
    puts $f "$net\troute_design -unroute -nets; update_clock_routing -net"
    close $f
    help update_clock_routing
    route_design -unroute -nets $net
    report_route_status -file $out/after_clock_unroute_status.rpt
    update_clock_routing -net $net
}

proc lp_stage {out} {
    set rows [lp_plan]
    lp_inventory $out/placement_before
    lp_cells $out/cells_before.tsv
    lp_clock_pins $out/clock_pins_before.tsv
    set log [open $out/placement_actions.tsv {WRONLY CREAT EXCL}]
    puts $log "cell\tprimitive\tbefore_loc\tbefore_bel\tafter_loc\tafter_bel"
    if {$::lp_mode eq "eco"} {
        # Validate the whole plan before moving any cell. No occupied BEL is evicted.
        foreach row $rows {
            lassign $row name primitive oldloc oldbel loc bel
            set cell [cp_one [get_cells $name] placement_cell]
            if {[get_property REF_NAME $cell] ne $primitive || [get_property LOC $cell] ne $oldloc || [get_property BEL $cell] ne $oldbel} {error "Placement donor differs: $name"}
            if {[get_property IS_LOC_FIXED $cell] || [get_property IS_BEL_FIXED $cell]} {error "Refusing fixed donor cell: $name"}
            set target [cp_one [get_bels $loc/$bel] target_bel]
            if {[llength [get_cells -quiet -of_objects $target]]} {error "Occupied destination: $target"}
        }
        foreach row $rows {
            lassign $row name primitive oldloc oldbel loc bel
            place_cell [list $name $loc/$bel]
            puts $log [join $row \t]
        }
        close $log
        report_route_status -file $out/after_moves_route_status.rpt
        lp_repair_clock $out
        report_route_status -file $out/after_clock_repair_route_status.rpt
        write_checkpoint $out/placed.dcp
        route_design -preserve -no_psir
        route_design -finalize
        write_checkpoint $out/routed.dcp
    } else {
        foreach row $rows {puts $log [join $row \t]};close $log
    }
    foreach row $rows {
        lassign $row name primitive oldloc oldbel loc bel
        set cell [get_cells $name]
        if {[get_property REF_NAME $cell] ne $primitive || [get_property LOC $cell] ne $loc || [lindex [split [get_property BEL $cell] .] end] ne $bel} {error "Final placement differs: $name"}
    }
    lp_inventory $out/placement_after
    lp_cells $out/cells_after.tsv
    lp_clock_pins $out/clock_pins_after.tsv
}

proc lp_run {checkpoint out mode} {
    if {[file exists $out] || $mode ni {probe eco reopen}} {error "Fresh output and implemented placement mode required"}
    set ::lp_source [file dirname [info script]]
    set ::lp_mode $mode
    if {$mode ne "probe"} {
        rename bi_pins lp_inherited_pins
        proc bi_pins {folder assign} {
            lp_inherited_pins $folder $assign
            if {$assign || [info exists ::lp_ran]} {error "Unexpected placement hook invocation"}
            set ::lp_ran 1
            lp_stage $folder
        }
        cp_run $checkpoint $out reopen
        if {![info exists ::lp_ran]} {error "Placement hook did not execute"}
        return
    }
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" || [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong placement donor"}
    foreach cmd {place_cell unplace_cell route_design get_bels} {help $cmd}
    write_xdc -constraints all $out/before.xdc
    wb_probe $out/registered_bready 1
    lp_inventory $out/placement
    report_route_status -file $out/route_status.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f BREADY_PLACEMENT_PROBE_EXPORTED;close $f
    close_design
}
