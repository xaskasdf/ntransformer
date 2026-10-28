# Isolated netlist integration of the functionally reviewed distribution stage.
set_param general.maxThreads 2
source [file join [file dirname [info script]] implement_hbm_board.tcl]
set ::ri_sync {core/reader/startup/memory_release_reg[2]}
set ::ri_dist core/reader/startup/memory_distribution_reg
set ::ri_ready {core/reader/startup/ready_meta_reg[2]/D}
proc ri_pin {name} {return [nc_one [get_pins $name] $name]}
proc ri_net {pin} {return [nc_one [get_nets -of_objects [ri_pin $pin]] net]}
proc ri_leaves {pin direction} {
    return [lsort -unique [get_pins -leaf -of_objects [get_nets -segments -of_objects [ri_pin $pin]] -filter "DIRECTION == $direction"]]
}
proc ri_save_list {path values} {
    set f [open $path {WRONLY CREAT EXCL}]
    foreach value $values {puts $f $value};close $f
}
proc ri_topology {out} {
    set a $::ri_sync;set b $::ri_dist
    set f [open $out/reset_topology.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    foreach name [list $a $b] {
        set cell [nc_one [get_cells $name] reset_cell]
        foreach k {REF_NAME INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED LOC BEL} {
            puts $f "$name\t$k\t[get_property $k $cell]"
        }
        foreach pin {C CE CLR D Q} {
            puts $f "$name/$pin\tDRIVERS\t[ri_leaves $name/$pin OUT]"
        }
        if {[get_property REF_NAME $cell] ne "FDCE" || [get_property INIT $cell] ne "1'b0" ||
            [get_property IS_CLR_INVERTED $cell] ne "1'b1"} {error "Reset primitive changed: $name"}
    }
    if {[get_property ASYNC_REG [get_cells $a]] ne "1" ||
        [get_property ASYNC_REG [get_cells $b]] in {1 TRUE true} ||
        [get_property DONT_TOUCH [get_cells $b]] in {1 TRUE true}} {error "Reset attributes changed"}
    set clear [nc_one [ri_leaves $b/CLR OUT] distribution_clear]
    if {$clear ne "$a/Q" || [get_property IS_INVERTED [ri_pin $b/CLR]] ne "1"} {error "Distribution clear source/polarity differs"}
    set clock [get_clocks -of_objects [ri_pin $b/C]]
    if {$clock ne "clk_out2_fabric_clocks" || [ri_leaves $a/C OUT] ne [ri_leaves $b/C OUT]} {error "Distribution clock differs"}
    puts $f "$b/C\tCLOCK\t$clock"
    foreach pin {D CE} {
        set driver [nc_one [ri_leaves $b/$pin OUT] constant]
        if {$driver ne [nc_one [ri_leaves $a/CE OUT] original_constant] ||
            [get_property REF_NAME [get_cells -of_objects [get_pins $driver]]] ne "VCC"} {error "Distribution constant differs"}
    }
    if {[ri_leaves $a/Q IN] ne "$b/CLR"} {error "Canonical reset has unexpected loads"}
    if {[nc_one [ri_leaves $::ri_ready OUT] readiness] ne "$b/Q"} {error "Readiness bypasses distribution"}
    puts $f "$a/Q\tLOADS\t[ri_leaves $a/Q IN]"
    puts $f "$::ri_ready\tDRIVERS\t[ri_leaves $::ri_ready OUT]"
    close $f
    ri_save_list $out/reset_consumers.txt [ri_leaves $b/Q IN]
    ri_save_list $out/cells.txt [lsort [get_cells -hier]]
    ri_save_list $out/nets.txt [lsort [get_nets -hier]]
}
proc ri_remap_constraints {out} {
    set f [open $out/before.xdc r];set original [read $f];close $f
    set old {set_max_delay -datapath_only -from [get_cells {core/reader/startup/memory_release_reg[2]}] -to [get_pins -quiet {core/reader/startup/ready_meta_reg[2]/D}] 4.000}
    set new {set_max_delay -datapath_only -from [get_cells core/reader/startup/memory_distribution_reg] -to [get_pins -quiet {core/reader/startup/ready_meta_reg[2]/D}] 4.000}
    set count 0;foreach line [split $original \n] {if {$line eq $old} {incr count}}
    if {$count!=1} {error "Expected one exact readiness constraint, got $count"}
    set f [open $out/remapped.xdc {WRONLY CREAT EXCL}]
    puts -nonewline $f [string map [list $old $new] $original];close $f
    # Vivado has no selective reset_path command. Reapply the complete exported
    # constraints with exactly the one source substitution; independently diff it.
    reset_timing
    read_xdc $out/remapped.xdc
    write_xdc -constraints all $out/remap_effective.xdc
}
proc ri_timing {out} {
    set f [open $out/distribution_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\tclock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach type {max min} {
        set path [nc_one [get_timing_paths -from [ri_pin $::ri_sync/C] -to [ri_pin $::ri_dist/CLR] -delay_type $type -max_paths 1] distribution_path]
        puts $f "[get_property STARTPOINT_PIN $path]\t[get_property ENDPOINT_PIN $path]\t$type\t[get_property STARTPOINT_CLOCK $path]\t[get_property ENDPOINT_CLOCK $path]\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]\t[get_property DATAPATH_DELAY $path]"
    };close $f
    report_timing -from [ri_pin $::ri_sync/C] -to [ri_pin $::ri_dist/CLR] -delay_type min_max -max_paths 2 -path_type full_clock_expanded -input_pins -file $out/distribution_recovery_removal.rpt
    report_timing -from [ri_pin $::ri_dist/C] -delay_type min_max -max_paths 8 -path_type full_clock_expanded -input_pins -file $out/distributed_reset_timing.rpt
}
proc ri_apb_aliases {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "net\texists\tleaf_inputs\tleaf_outputs\ttop_ports"
    foreach stack {0 1} {
        foreach signal {PREADY PSLVERR} {
            foreach name [list hbm_binding/memory/inst/APB_${stack}_${signal} \
                hbm_binding/memory/inst/TWO_STACK.u_hbm_top/APB_${stack}_${signal} \
                hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_apb_arbiter_${stack}/extrnl_apb_[string tolower $signal]] {
                set net [get_nets -quiet $name]
                set inputs {};set outputs {};set ports {}
                if {[llength $net]} {
                    nc_one $net apb_alias
                    set segments [get_nets -segments $name]
                    set inputs [lsort -unique [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]]
                    set outputs [lsort -unique [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]]
                    set ports [lsort -unique [get_ports -quiet -of_objects $segments]]
                }
                puts $f "$name\t[llength $net]\t$inputs\t$outputs\t$ports"
            }
        }
    };close $f
}
proc ri_calibration_selectors {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "lane\tfamily\tcell"
    for {set lane 0} {$lane<16} {incr lane} {
        foreach family {testclk_cnt_reg freq_cnt_o_reg} {
            set expression [format {NAME =~ *gen_cpll_cal_inst[%d].*U_TXOUTCLK_FREQ_COUNTER/%s*} $lane $family]
            set cells [lsort [get_cells -hierarchical -filter $expression]]
            if {![llength $cells]} {error "Empty calibration selector $lane $family"}
            foreach cell $cells {puts $f "$lane\t$family\t$cell"}
        }
    };close $f
}
proc ri_run {checkpoint out mode {alias_reference ""}} {
    if {[file exists $out] || $mode ni {edit route reopen}} {error "Fresh output and supported reset integration mode required"}
    set source [file dirname [info script]]
    file mkdir $out
    if {$alias_reference ne ""} {
        if {$mode ne "reopen"} {error "Alias reference is read-only reopen evidence"}
        open_checkpoint $alias_reference
        if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
            [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong alias reference"}
        set f [open $out/alias_reference_sha256.txt {WRONLY CREAT EXCL}]
        puts $f [lindex [exec sha256sum $alias_reference] 0];close $f
        ri_apb_aliases $out/reference_aliases.tsv
        ri_calibration_selectors $out/reference_calibration_selectors.tsv
        close_design
    }
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong integrated board donor"}
    write_xdc -constraints all $out/before.xdc
    bi_pins $out 0
    if {$mode eq "edit"} {
        set a $::ri_sync;set b $::ri_dist
        if {[llength [get_cells -quiet $b]]} {error "Donor already contains distribution"}
        set consumers [ri_leaves $a/Q IN]
        if {[llength $consumers]!=1091} {error "Unexpected donor reset fanout"}
        set cells [lsort [get_cells -hier]];set nets [lsort [get_nets -hier]]
        ri_save_list $out/original_consumers.txt $consumers
        ri_save_list $out/original_cells.txt $cells
        ri_save_list $out/original_nets.txt $nets
        set output_net [ri_net $a/Q];set clock_net [ri_net $a/C];set constant_net [ri_net $a/CE]
        create_cell -reference FDCE $b
        set_property INIT 1'b0 [get_cells $b]
        set_property IS_CLR_INVERTED 1'b1 [get_cells $b]
        set new_net core/reader/startup/memory_sync_release
        create_net $new_net
        disconnect_net -net $output_net -objects [ri_pin $a/Q]
        connect_net -net $new_net -objects [get_pins [list $a/Q $b/CLR]]
        connect_net -net $output_net -objects [ri_pin $b/Q]
        connect_net -net $clock_net -objects [ri_pin $b/C]
        connect_net -net $constant_net -objects [get_pins [list $b/D $b/CE]]
        # Save the netlist edit before constraint/audit operations can fail.
        write_checkpoint $out/edited_unmapped.dcp
        ri_remap_constraints $out
        write_checkpoint $out/edited.dcp
        ri_topology $out
        if {[ri_leaves $b/Q IN] ne $consumers ||
            [lsort [get_cells -hier]] ne [lsort [concat $cells [list $b]]] ||
            [lsort [get_nets -hier]] ne [lsort [concat $nets [list $new_net]]]} {error "Edit exceeds one register and one net"}
        set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
        puts $f HBM_RESET_BOARD_EDIT_EXPORTED;close $f
    } else {
        report_timing_summary -delay_type min_max -file $out/timing_before.rpt
        if {$mode eq "route"} {
            # The -eco placer/router switches target Versal. For this UltraScale+
            # part, explicitly reuse the edited donor's retained physical data.
            read_checkpoint -incremental $checkpoint
            report_incremental_reuse -file $out/reuse_before.rpt
            place_design -no_psip
            write_checkpoint $out/placed.dcp
            report_incremental_reuse -file $out/reuse_placed.rpt
            route_design -no_psir
            write_checkpoint $out/routed.dcp
            report_incremental_reuse -file $out/reuse_routed.rpt
        }
        ri_topology $out
        ri_timing $out
        ri_apb_aliases $out/current_aliases.tsv
        ri_calibration_selectors $out/current_calibration_selectors.tsv
        bi_reports $out/reports
        bi_audit $source $out/contracts
        set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
        puts $f HBM_RESET_BOARD_${mode}_EXPORTED;close $f
    }
    close_design
}
