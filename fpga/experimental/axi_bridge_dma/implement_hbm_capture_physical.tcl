# Fresh full-RTL candidate; apply the frozen contracts, then reuse placement.
set_param general.maxThreads 2
source [file join [file dirname [info script]] integrate_hbm_reset_distribution.tcl]
source [file join [file dirname [info script]] audit_hbm_capture_physical.tcl]
source [file join [file dirname [info script]] inspect_hbm_routed_timing.tcl]

# Fresh synthesis can leave the cell property unset while the actual CLR pin
# reports IS_INVERTED=0. Preserve the raw value and prove the explicit inverter;
# do not write properties or change the netlist to satisfy an audit.
rename bi_readiness_clear cp_inherited_readiness_clear
proc bi_readiness_clear {out source clr} {
    set raw [get_property IS_CLR_INVERTED [get_cells $source]]
    if {$raw ne ""} {return [cp_inherited_readiness_clear $out $source $clr]}
    set canonical [nc_one [bc_driver [nb_pin core/management_cold_rst_n]] canonical_reset]
    set pin_inverted [get_property IS_INVERTED [nb_pin $source/CLR]]
    set driver [get_cells -of_objects [get_pins $clr]]
    if {[get_property REF_NAME $driver] ne "LUT1" || [get_property INIT $driver] ne "2'h1" ||
        $pin_inverted ne "0" || [nc_one [bc_driver [nb_pin $driver/I0]] inverter_source] ne $canonical} {
        error "Unset cell inversion is not the proven explicit-inverter case"
    }
    set f [open $out/readiness_reset.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tclear_driver\tcanonical_driver\tencoding\tclear_inverted\tpin_inverted\tdriver_reference\tdriver_init\tclear_inverted_raw\tinversion_basis"
    puts $f "$source\t$clr\t$canonical\tEXPLICIT_INVERTER\t0\t0\tLUT1\t2'h1\t\tPIN_IS_INVERTED_WITH_VERIFIED_LUT"
    close $f
    return $canonical
}

proc cp_driver {pin} {
    return [cp_one [get_pins -leaf -of_objects [get_nets -of_objects [cp_one [get_pins $pin] $pin]] -filter {DIRECTION == OUT}] driver]
}
proc cp_reset {out timed} {
    file mkdir $out
    set sync {core/reader/startup/memory_release_reg[2]}
    set dist core/reader/startup/memory_distribution_reg
    set f [open $out/reset.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    foreach name [list $sync $dist] {
        set cell [cp_one [get_cells $name] reset_register]
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0"} {error "Cold-release primitive changed"}
        foreach key {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED LOC BEL} {
            puts $f "$name\t$key\t[get_property $key $cell]"
        }
        foreach pin {C D CE CLR} {puts $f "$name/$pin\tDRIVER\t[cp_driver $name/$pin]"}
    }
    if {[get_property ASYNC_REG [get_cells $sync]] ne "1" ||
        [get_property ASYNC_REG [get_cells $dist]] in {1 TRUE true} ||
        [get_property DONT_TOUCH [get_cells $dist]] in {1 TRUE true}} {error "Reset distribution attributes changed"}
    foreach name [list $sync $dist] {
        set clock [get_clocks -of_objects [get_pins $name/C]]
        if {$clock ne "clk_out2_fabric_clocks"} {error "Reset release clock changed"}
        puts $f "$name/C\tCLOCK\t$clock"
    }
    foreach pin {D CE} {
        set driver [cp_driver $dist/$pin]
        if {[get_property REF_NAME [get_cells -of_objects [get_pins $driver]]] ne "VCC"} {error "Distribution data/enable changed"}
    }
    set driver [cp_driver $dist/CLR]
    set inverted [get_property IS_INVERTED [get_pins $dist/CLR]]
    puts $f "$dist/CLR\tIS_INVERTED\t$inverted"
    if {$driver eq "$sync/Q"} {
        if {$inverted ne "1"} {error "Wrong primitive clear polarity"}
        set sole $dist/CLR
        puts $f "$dist\tRESET_ENCODING\tPRIMITIVE_INVERTED_CLR"
    } else {
        set inv [get_cells -of_objects [get_pins $driver]]
        if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1" ||
            $inverted ne "0" || [cp_driver $inv/I0] ne "$sync/Q"} {error "Wrong reset inverter"}
        set loads [ri_leaves $driver IN]
        if {$loads ne "$dist/CLR"} {error "Extra reset inverter load"}
        puts $f "$inv\tREF_NAME\tLUT1"
        puts $f "$inv\tINIT\t2'h1"
        puts $f "$inv/I0\tDRIVER\t$sync/Q"
        puts $f "$inv/O\tLOADS\t$loads"
        puts $f "$dist\tRESET_ENCODING\tEXPLICIT_INVERTER"
        set sole $inv/I0
    }
    set consumers [ri_leaves $sync/Q IN]
    if {$consumers ne $sole} {error "Synchronizer has collateral consumers"}
    puts $f "$sync/Q\tLOADS\t$consumers"
    set ready {core/reader/startup/ready_meta_reg[2]/D}
    if {[cp_driver $ready] ne "$dist/Q"} {error "Readiness bypasses distribution"}
    puts $f "$ready\tDRIVER\t$dist/Q"
    close $f
    ri_save_list $out/distribution_loads.txt [ri_leaves $dist/Q IN]
    if {$timed} {
        foreach type {max min} {
            ht_table $out recovery_$type [get_timing_paths -from [get_pins $sync/C] -to [get_pins $dist/CLR] -delay_type $type -max_paths 1]
        }
    }
}

proc cp_run {checkpoint out mode {incremental ""}} {
    if {[file exists $out] || $mode ni {constrain route reopen}} {error "Fresh output and supported mode required"}
    set source [file dirname [info script]]
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong full-board candidate"}
    write_xdc -constraints all $out/before.xdc
    if {$mode eq "constrain"} {
        set apply $out/applied;file mkdir $apply
        nc_clock_map $apply
        nc_constrain $apply 1
        nc_topology $apply
        nc_paths $apply
        file rename $apply/raw_lock.tsv $apply/inherited_raw_lock.tsv
        bc_structure $apply
        bc_snapshot $apply 1
        wp_connections $apply
        wp_bounds $apply 1
        wp_sync $apply
        bd_reset $apply 1
        bd_lock $apply 1
        bi_pins $out 1
        write_xdc -constraints all $out/applied.xdc
        write_checkpoint $out/bounded.dcp
        opt_design
        write_checkpoint $out/optimized.dcp
        cp_probe $out/response_capture 0
        cp_reset $out/reset 0
    } else {
        bi_pins $out 0
        if {$mode eq "route"} {
            if {$incremental eq ""} {error "Exact incremental donor required"}
            set f [open $out/incremental_sha256.txt {WRONLY CREAT EXCL}]
            puts $f [lindex [exec sha256sum $incremental] 0];close $f
            read_checkpoint -incremental $incremental
            report_incremental_reuse -file $out/reuse_before.rpt
            place_design -no_psip
            write_checkpoint $out/placed.dcp
            report_incremental_reuse -file $out/reuse_placed.rpt
            route_design -no_psir
            write_checkpoint $out/routed.dcp
            report_incremental_reuse -file $out/reuse_routed.rpt
        }
        cp_probe $out/response_capture 1
        cp_reset $out/reset 1
        ht_table $out worst_setup [get_timing_paths -delay_type max -max_paths 8 -nworst 1]
        set ready_ends {}
        for {set channel 0} {$channel<2} {incr channel} {
            set pin [cp_one [get_pins [format {core/reader/channels[%d].adapter/h_rready} $channel]] rready]
            set ends [all_fanout -flat -endpoints_only -from $pin]
            if {![llength $ends]} {error "Missing HBM RREADY endpoints"}
            foreach to $ends {lappend ready_ends $to}
        }
        ht_table $out rready_max [get_timing_paths -to $ready_ends -delay_type max -max_paths 8 -nworst 1]
        ht_table $out rready_min [get_timing_paths -to $ready_ends -delay_type min -max_paths 8 -nworst 1]
    }
    ri_calibration_selectors $out/calibration_selectors.tsv
    bi_reports $out/reports
    bi_audit $source $out/contracts
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f CAPTURE_BOARD_${mode}_EXPORTED;close $f
    close_design
}
