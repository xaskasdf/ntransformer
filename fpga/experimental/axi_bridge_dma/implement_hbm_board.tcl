# Private OOC implementation of the sealed board CDC composition.
set_param general.maxThreads 2
source [file join [file dirname [info script]] audit_hbm_board_cdc.tcl]

proc bi_readiness_clear {out source clr} {
    set canonical [nc_one [bc_driver [nb_pin core/management_cold_rst_n]] canonical_reset]
    set inverted [get_property IS_CLR_INVERTED [get_cells $source]]
    set pin_inverted [get_property IS_INVERTED [nb_pin $source/CLR]]
    set driver [get_cells -of_objects [get_pins $clr]]
    set ref [get_property REF_NAME $driver]
    set init [get_property INIT $driver]
    if {$clr eq $canonical && $ref eq "FDCE" && $inverted in {1 1'b1 TRUE true} && $pin_inverted eq "1"} {
        set encoding PRIMITIVE_INVERTED_CLR
    } elseif {$ref eq "LUT1" && $init eq "2'h1" && $inverted in {0 1'b0 FALSE false} && $pin_inverted eq "0" &&
              [nc_one [bc_driver [nb_pin $driver/I0]] reset_inverter_input] eq $canonical} {
        set encoding EXPLICIT_INVERTER
    } else {error "Readiness reset no longer implements active-low canonical release"}
    set f [open $out/readiness_reset.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tclear_driver\tcanonical_driver\tencoding\tclear_inverted\tpin_inverted\tdriver_reference\tdriver_init"
    puts $f "$source\t$clr\t$canonical\t$encoding\t$inverted\t$pin_inverted\t$ref\t$init";close $f
    return $canonical
}

proc bi_pins {out assign} {
    set spec {
        sys_clk_p AR40 MGTREFCLK0P_125 125 NONE
        sys_clk_n AR41 MGTREFCLK0N_125 125 NONE
        sys_rst_n BF41 IO_T3U_N12_PERSTN0_65 65 LVCMOS18
        fabric_ref_clk_in BK31 IO_L11P_T1U_N8_GC_64 64 LVCMOS18
        mgmt_clk_p BJ33 IO_L14P_T2L_N2_GC_64 64 LVDS
        mgmt_clk_n BJ34 IO_L14N_T2L_N3_GC_64 64 LVDS
        hbm_ref0_clk_p BK33 IO_L12P_T1U_N10_GC_64 64 LVDS
        hbm_ref0_clk_n BL33 IO_L12N_T1U_N11_GC_64 64 LVDS
        hbm_ref1_clk_p BH32 IO_L13P_T2L_N0_GC_QBC_64 64 LVDS
        hbm_ref1_clk_n BJ32 IO_L13N_T2L_N1_GC_QBC_64 64 LVDS
        supervisor_rst_n_pad BG33 IO_T2U_N12_64 64 LVCMOS18
    }
    if {[llength [get_ports]]!=75} {error "Board must expose exactly 75 bits"}
    if {$assign} {
        foreach {name ball function bank standard} $spec {
            if {[get_property PACKAGE_PIN [get_ports $name]] ne ""} {error "Donor already assigns $name"}
        }
    }
    set f [open $out/trial_pins.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tball\tfunction\tbank\tstandard\tslr\tstatus"
    foreach {name ball function bank standard} $spec {
        set pin [nc_one [get_package_pins $ball] package_pin]
        if {[get_property PIN_FUNC $pin] ne $function || [get_property BANK $pin] ne $bank ||
            ![get_property IS_BONDED $pin] || [get_property PIN_RESTRICTIONS $pin] ne ""} {error "Package mismatch $ball"}
        if {$assign} {
            set_property PACKAGE_PIN $ball [get_ports $name]
            if {$standard ne "NONE"} {set_property IOSTANDARD $standard [get_ports $name]}
        }
        if {[get_property PACKAGE_PIN [get_ports $name]] ne $ball ||
            ($standard ne "NONE" && [get_property IOSTANDARD [get_ports $name]] ne $standard)} {error "Pin assignment differs $name"}
        set slr [lsort -unique [get_slrs -of_objects [get_sites -of_objects $pin]]]
        if {$bank eq "64" && $slr ne "SLR0"} {error "Bank 64 outside HBM SLR"}
        if {[string match *_clk_p $name] && $name ne "sys_clk_p" && ![get_property IS_GLOBAL_CLK $pin]} {error "Reference not GCIO"}
        puts $f "$name\t$ball\t$function\t$bank\t$standard\t$slr\tEXPERIMENT_ONLY"
    };close $f
    foreach {a b} {AR40 AR41 BK33 BL33 BH32 BJ32 BJ33 BJ34} {
        if {[get_property DIFF_PAIR_PIN [get_package_pins $a]] ne $b || [get_property DIFF_PAIR_PIN [get_package_pins $b]] ne $a} {error "Wrong differential pair $a $b"}
    }
    foreach bank {64 65} {
        if {[get_property BANK_TYPE [get_iobanks $bank]] ne "BT_HIGH_PERFORMANCE"} {error "Wrong I/O bank type"}
    }
    set site [nc_one [get_sites -of_objects [get_package_pins AR40] -filter {SITE_TYPE == GTYE4_COMMON}] pcie_ref]
    if {$site ne "GTYE4_COMMON_X0Y1"} {error "Wrong reference common site"}
    if {$assign} {set_property LOC $site [get_cells refclk_ibuf]}
    if {[get_property LOC [get_cells refclk_ibuf]] ne $site} {error "Reference LOC missing"}
}

proc bi_synchronous {source out} {
    set input [open $source/expected_synchronous.tsv r]
    gets $input header
    set f [open $out/synchronous_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tfrom\tto\ttype\tclock\trequirement_ns\tslack_ns\tstatus"
    while {[gets $input line]>=0} {
        lassign [split $line "\t"] kind from to type clock bound old_slack
        set src [get_pins -quiet $from];set dst [get_pins -quiet $to]
        if {![llength $src] || ![llength $dst]} {
            if {$to ni {board_startup/lock_fault_reg/D board_startup/lock_fault_reg/CLR} ||
                [llength [get_cells -quiet board_startup/lock_fault_reg]]} {error "Required synchronous endpoint removed: $from $to"}
            puts $f "$kind\t$from\t$to\t$type\t$clock\t$bound\t\tPRUNED_UNUSED_LOCK_FAULT"
            continue
        }
        set path [nc_one [get_timing_paths -quiet -from $src -to $dst -delay_type $type -max_paths 1] synchronous_path]
        if {[get_property STARTPOINT_CLOCK $path] ne $clock || [get_property ENDPOINT_CLOCK $path] ne $clock ||
            abs([get_property REQUIREMENT $path]-$bound)>0.001} {error "Synchronous timing contract changed: $from $to"}
        puts $f "$kind\t$from\t$to\t$type\t$clock\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]\tTIMED"
    };close $f;close $input
}

proc bi_audit {source out} {
    file mkdir $out
    # These procedures only query existing constraints (apply=0).
    nc_clock_map $out
    nc_paths $out
    bc_snapshot $out 0
    wp_connections $out
    wp_bounds $out 0
    wp_sync $out
    bd_lock $out 0
    bi_synchronous $source $out
    ro_audit $out
    set f [open $out/AUDIT_STATUS.txt {WRONLY CREAT EXCL}]
    puts $f BOARD_IMPLEMENTATION_CONTRACTS_EXPORTED_REVIEW_REQUIRED;close $f
}

proc bi_reports {out} {
    file mkdir $out
    write_xdc -constraints all $out/effective.xdc
    report_drc -file $out/drc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored.rpt
    report_utilization -hierarchical -file $out/utilization.rpt
    report_methodology -file $out/methodology.rpt
    report_waivers -file $out/waivers.rpt
    report_route_status -file $out/route_status.rpt
    report_io -file $out/io.rpt
}

proc bi_run {checkpoint out mode} {
    if {[file exists $out] || $mode ni {optimize audit place route reopen}} {error "Fresh output and valid mode required"}
    set source [file dirname [info script]]
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong board checkpoint"}
    write_xdc -constraints all $out/before.xdc
    bi_pins $out [expr {$mode eq "optimize"}]
    if {$mode eq "optimize"} {
        write_xdc -constraints all $out/pinned.xdc
        write_checkpoint $out/pinned.dcp
        report_drc -file $out/pinned_drc.rpt
        opt_design
        set stage optimized
    } elseif {$mode eq "place"} {
        place_design
        set stage placed
    } elseif {$mode eq "route"} {
        route_design
        set stage routed
    } elseif {$mode eq "audit"} {set stage audited
    } else {set stage reopened}
    if {$mode ni {audit reopen}} {write_checkpoint $out/$stage.dcp}
    # Save stage state and diagnostics even when a structural guard then fails.
    bi_reports $out/reports
    bi_audit $source $out/contracts
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f BOARD_IMPLEMENTATION_${stage}_EXPORTED_REVIEW_REQUIRED;close $f
    close_design
}
