# Bounded physical experiment on one HBM response enable; no RTL/clock edits.
set_param general.maxThreads 2
source [file join [file dirname [info script]] integrate_hbm_reset_distribution.tcl]
source [file join [file dirname [info script]] inspect_hbm_routed_timing.tcl]
set ::he_prefix {core/reader/channels[0].adapter/tracker/response_cut}
set ::he_driver {core/reader/channels[0].adapter/tracker/response_cut/slot1[264]_i_1}

proc he_drivers {pin} {
    # -leaf traverses hierarchical nets directly (UG894). Do not expand every
    # alias with -segments or repeatedly resolve typed objects by string name.
    set net [get_nets -quiet -of_objects $pin]
    if {![llength $net]} {return {}}
    nc_one $net input_pin_net
    set key [get_property NAME $net]
    if {![info exists ::he_driver_cache($key)]} {
        set ::he_driver_cache($key) [lsort -unique [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]]
    }
    return $::he_driver_cache($key)
}

proc he_probe {out} {
    # Discard every cached association on each snapshot/design boundary.
    array unset ::he_driver_cache
    file mkdir $out
    ri_topology $out
    set ce {};set drivers {}
    set f [open $out/enable_loads.tsv {WRONLY CREAT EXCL}]
    puts $f "pin\tdriver\tcell_type\tlocation\tbel"
    for {set bit 0} {$bit<265} {incr bit} {
        set pin [format {%s/slot1_reg[%d]/CE} $::he_prefix $bit]
        set p [ri_pin $pin];lappend ce $pin
        set d [nc_one [he_drivers $p] enable_driver]
        lappend drivers [get_property NAME [get_cells -of_objects [get_pins $d]]]
        set cell [get_cells -of_objects $p]
        puts $f "$pin\t$d\t[get_property REF_NAME $cell]\t[get_property LOC $cell]\t[get_property BEL $cell]"
    };close $f
    set f [open $out/driver_logic.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tproperty\tvalue"
    foreach name [lsort -unique $drivers] {
        set cell [nc_one [get_cells $name] driver_cell]
        foreach k {REF_NAME INIT LOC BEL DONT_TOUCH PHYS_OPT_MODIFIED} {
            puts $f "$name\t$k\t[get_property $k $cell]"
        }
        foreach p [get_pins -of_objects $cell] {
            set direction [get_property DIRECTION $p]
            set leaf [get_property REF_PIN_NAME $p]
            if {$direction eq "IN"} {
                puts $f "$name\t$leaf\t[he_drivers $p]"
            } else {
                puts $f "$name\t${leaf}_LOADS\t[ri_leaves $p IN]"
            }
        }
    };close $f
    # Logical names/properties for the whole target tracker detect collateral edits.
    set f [open $out/tracker_logic.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tproperty\tvalue"
    set tracker {core/reader/channels[0].adapter/tracker/}
    set matched 0
    foreach cell [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
        set name [get_property NAME $cell]
        # Literal prefix: Tcl glob brackets would misinterpret channels[0].
        if {[string first $tracker $name]!=0} {continue}
        incr matched
        set properties [list_property $cell]
        foreach k {REF_NAME INIT ASYNC_REG DONT_TOUCH} {
            if {$k in $properties} {puts $f "$name\t$k\t[get_property $k $cell]"}
        }
        foreach p [get_pins -of_objects $cell -filter {DIRECTION == IN}] {
            puts $f "$name\tPIN_[get_property REF_PIN_NAME $p]\t[he_drivers $p]"
        }
    };close $f
    if {$matched<530} {error "Incomplete response tracker connectivity export"}
    ht_table $out enable_max [get_timing_paths -to [get_pins $ce] -delay_type max -max_paths 265 -nworst 1]
    ht_table $out enable_min [get_timing_paths -to [get_pins $ce] -delay_type min -max_paths 265 -nworst 1]
    ht_table $out worst_setup [get_timing_paths -delay_type max -max_paths 8 -nworst 1]
    report_route_status -file $out/route.rpt
    report_timing_summary -delay_type min_max -file $out/timing.rpt
    write_xdc -constraints all $out/effective.xdc
}

proc he_run {checkpoint out mode {reference ""}} {
    if {[file exists $out] || $mode ni {placement reopen}} {error "Fresh response-enable output/mode required"}
    set source [file dirname [info script]]
    file mkdir $out
    if {$mode eq "reopen"} {
        if {$reference eq ""} {error "Exact original donor required for semantic reopen"}
        set f [open $out/reference_sha256.txt {WRONLY CREAT EXCL}]
        puts $f [lindex [exec sha256sum $reference] 0];close $f
        open_checkpoint $reference
        he_probe $out/reference
        close_design
    }
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong response-enable checkpoint"}
    help phys_opt_design
    write_xdc -constraints all $out/before.xdc
    bi_pins $out 0
    he_probe $out/before
    set f [open $out/experiment.tsv {WRONLY CREAT EXCL}]
    puts $f "mode\tcommand\ttarget\tclock_or_exception_edits"
    if {$mode eq "placement"} {
        set driver [nc_one [get_cells $::he_driver] original_enable_driver]
        if {[get_property REF_NAME $driver] ne "LUT4" ||
            [llength [ri_leaves $::he_driver/O IN]]!=265} {error "Unexpected enable cone"}
        set target [nc_one [get_path_groups clk_out2_fabric_clocks] memory_path_group]
        puts $f "placement\tphys_opt_design -placement_opt -routing_opt -path_groups\t$target\t0";close $f
        # Forced replication is not supported by this tool in post-route mode.
        # Explicit physical-only options disable the default logical transforms.
        phys_opt_design -placement_opt -routing_opt -path_groups $target
        write_checkpoint $out/routed.dcp
    } else {
        puts $f "reopen\topen_checkpoint\t$checkpoint\t0";close $f
    }
    he_probe $out/after
    ri_timing $out
    ri_calibration_selectors $out/calibration_selectors.tsv
    bi_reports $out/reports
    bi_audit $source $out/contracts
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_RESPONSE_ENABLE_${mode}_EXPORTED_REVIEW_REQUIRED;close $f
    close_design
}
