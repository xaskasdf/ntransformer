# One-net route experiment; the frozen board and BREADY audits remain active.
source [file join [file dirname [info script]] inspect_hbm_bready_balanced_placement.tcl]

proc lr_plan {} {
    set f [open [file join $::lp_source route_plan.tsv] r]
    set lines [split [string trim [read $f]] \n];close $f
    if {[llength $lines]!=2 || [lindex $lines 0] ne "driver_pin\tsink_pin\tnet\tmethod"} {error "One-net route plan required"}
    set row [split [lindex $lines 1] \t]
    if {$row ne [list {weight_loader/active_reg[25]/Q} {weight_loader/address_reg[25]/D} {weight_loader/active_reg_n_0_[25]} delay]} {error "Unexpected route target"}
    return $row
}

proc lr_net_inventory {path} {
    set names [open ${path}.names.tsv {WRONLY CREAT EXCL}];puts $names net
    foreach net [get_nets -hier] {puts $names $net};close $names
    set helper [file join $::lp_source hash_hbm_route_stream.py]
    set f [open [list | python3 -B $helper $path] w]
    fconfigure $f -encoding utf-8 -translation lf
    # Cover every canonical route once, plus the full logical name inventory.
    foreach net [get_nets -hier -top_net_of_hierarchical_group] {
        set route [get_property ROUTE $net]
        regsub -all {\s+} $route { } route
        puts $f [join [list $net [string trim $route] [get_property ROUTE_STATUS $net] [get_property IS_ROUTE_FIXED $net]] \t]
    }
    close $f
}

proc lr_target {out net} {
    file mkdir $out
    lassign [lr_plan] driver sink name method
    set f [open $out/pins.tsv {WRONLY CREAT EXCL}];puts $f "net\tpin\tdirection"
    foreach pin_name [lsort [get_pins -leaf -of_objects $net]] {
        set pin [get_pins $pin_name]
        puts $f [join [list $net $pin [get_property DIRECTION $pin]] \t]
    }
    close $f
    rp_timing $out slow_max {weight_loader/active_reg[25]/C} $sink max
}

proc lp_stage {out} {
    if {$::lp_mode ni {eco reopen}} {error "Route mode must be eco or reopen"}
    lassign [lr_plan] driver sink name method
    set net [cp_one [get_nets -of_objects [get_pins $driver]] loader_route_net]
    if {$net ne $name || [rp_driver $sink] ne $driver || [get_property IS_ROUTE_FIXED $net]} {error "Loader donor route binding differs"}
    lp_cells $out/cells_before.tsv
    sp_inventory $out/placement_before
    lp_clock_pins $out/clock_pins_before.tsv
    sp_clock_route $out/clock_pips_before.tsv
    lr_net_inventory $out/net_routes_before.tsv
    lr_target $out/target_before $net
    set f [open $out/route_actions.tsv {WRONLY CREAT EXCL}]
    puts $f "net\tcommand"
    if {$::lp_mode eq "eco"} {
        puts $f "$net\troute_design -unroute -nets; route_design -nets -delay; route_design -finalize"
        close $f
        route_design -unroute -nets $net
        route_design -nets $net -delay
        route_design -finalize
        write_checkpoint $out/routed.dcp
    } else {close $f}
    # Reopen is read-only: one independent inventory is sufficient to compare
    # against the ECO's final inventory. Do not fabricate a second snapshot.
    if {$::lp_mode eq "eco"} {lr_net_inventory $out/net_routes_after.tsv}
    lr_target $out/target_after $net
    lp_cells $out/cells_after.tsv
    sp_inventory $out/placement_after
    lp_clock_pins $out/clock_pins_after.tsv
    sp_clock_route $out/clock_pips_after.tsv
}
