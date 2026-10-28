# Rebind the validated native groups and audit the new board reset boundary.
source [file join [file dirname [info script]] audit_hbm_loader_readiness.tcl]
source [file join [file dirname [info script]] audit_hbm_reference_ownership.tcl]

proc bd_paths {f label from to clock period} {
    foreach type {max min} bound [list $period 0.0] {
        set path [nc_one [get_timing_paths -quiet -from $from -to $to -delay_type $type -max_paths 1] "$label $type"]
        if {[get_property STARTPOINT_CLOCK $path] ne $clock || [get_property ENDPOINT_CLOCK $path] ne $clock ||
            abs([get_property REQUIREMENT $path]-$bound)>0.001} {error "Cut or changed synchronous path: $label $from $to"}
        puts $f [join [list $label $from $to $type $clock [get_property REQUIREMENT $path] [get_property SLACK $path]] "\t"]
    }
}

proc bd_census {out} {
    set f [open $out/board_registers.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\treference\tclock\tasync_reg"
    foreach cell [lsort [get_cells -hier -filter {NAME =~ board_startup/* && IS_SEQUENTIAL == 1}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_clocks -of_objects [get_pins $cell/C]]\t[get_property ASYNC_REG $cell]"
    };close $f
    set f [open $out/board_reset_fanout.tsv {WRONLY CREAT EXCL}]
    puts $f "source\tendpoint\treference\tclock\tasync_reg\tdirect_driver"
    foreach name {supervisor_rst_n_pad board_startup/configuration_rst_n_reg/Q board_startup/release_sync_reg[2]/Q core/reader/startup/source_release_reg[2]/Q} {
        if {$name eq "supervisor_rst_n_pad"} {set src [get_ports $name]} else {set src [nb_pin $name]}
        foreach to [lsort [all_fanout -flat -endpoints_only -from $src]] {
            set cell [get_cells -quiet -of_objects $to]
            if {[llength $cell]!=1} {error "Unexpected reset endpoint $to"}
            puts $f "$name\t$to\t[get_property REF_NAME $cell]\t[get_clocks -quiet -of_objects [get_pins -quiet $cell/C]]\t[get_property ASYNC_REG $cell]\t[bc_driver $to]"
        }
    };close $f
}

proc bd_reset {out apply} {
    set mgmt [dict get $::nc_clocks mgmt_clk]
    set timed [open $out/board_synchronous_paths.tsv {WRONLY CREAT EXCL}]
    puts $timed "kind\tfrom\tto\ttype\tclock\trequirement_ns\tslack_ns"
    set exceptions [open $out/board_reset_exceptions.tsv {WRONLY CREAT EXCL}]
    puts $exceptions "kind\tfrom\tto\tend_clock"
    # Only raw supervisor assertion/release at the three reset synchronizer
    # controls is excepted. All subsequent management recovery/removal stays timed.
    set supervisor [nc_one [get_ports supervisor_rst_n_pad] supervisor]
    set expected {}
    foreach bit {0 1 2} {lappend expected [format {board_startup/release_sync_reg[%d]/CLR} $bit]}
    if {[lsort [all_fanout -flat -endpoints_only -from $supervisor]] ne [lsort $expected]} {error "Supervisor escapes release synchronizer"}
    foreach to $expected {
        set pin [nb_pin $to]
        if {$apply} {set_false_path -from $supervisor -to $pin}
        puts $exceptions "supervisor\t$supervisor\t$pin\t$mgmt"
    }
    set sf [open $out/board_synchronizers.tsv {WRONLY CREAT EXCL}]
    puts $sf "cell\tclock\tasync_reg"
    set cells [get_cells -hier -quiet -filter {NAME =~ board_startup/* && ASYNC_REG == TRUE}]
    if {[llength $cells]!=6} {error "Expected six board synchronizer registers"}
    foreach chain {release_sync locked_sync} {
        foreach bit {0 1 2} {
            set cell [wp_cell [format {board_startup/%s_reg[%d]} $chain $bit]]
            if {[lsearch -exact $cells $cell]<0} {error "Board synchronizer marking changed"}
            nc_clock $cell mgmt_clk
            puts $sf "$cell\t$mgmt\t[get_property ASYNC_REG $cell]"
            if {$bit==2} {continue}
            set next [nb_pin [format {board_startup/%s_reg[%d]/D} $chain [expr {$bit+1}]]]
            set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects [nb_pin $cell/Q]] -filter {DIRECTION == IN}]
            if {[nc_one $loads board_interstage_load] ne $next} {error "Extra board interstage consumer $cell: $loads"}
            bd_paths $timed interstage [nb_pin $cell/C] $next $mgmt 10.0
        }
    };close $sf
    # The board-local synchronized release drives only same-domain async
    # controls. Require recovery/removal timing on every one, including LOCKED.
    set from [nb_pin {board_startup/release_sync_reg[2]/C}]
    set endpoints [all_fanout -flat -endpoints_only -from [nb_pin {board_startup/release_sync_reg[2]/Q}]]
    if {[llength $endpoints]!=15} {error "Board release fanout differs: $endpoints"}
    foreach to $endpoints {
        if {![regexp {^board_startup/.+/(CLR|PRE)$} $to]} {error "Board release bypass: $to"}
        bd_paths $timed board_recovery $from $to $mgmt 10.0
    }
    # One cold-release chain per destination domain, with no direct raw-reset
    # fanout into the reader status synchronizers after the RTL correction.
    set cfg [nb_pin board_startup/configuration_rst_n_reg/C]
    set expected {}
    foreach chain {core/endpoint/guard/cold_m core/endpoint/guard/cold_a core/reader/startup/source_release core/reader/startup/memory_release} domain {mgmt_clk pcie_clk reader_clk mem_clk} {
        foreach bit {0 1 2} {
            set cell [wp_cell [format {%s_reg[%d]} $chain $bit]]
            nc_clock $cell $domain
            if {[get_property ASYNC_REG $cell] ni {1 TRUE true}} {error "Unmarked cold release $cell"}
            set to [nb_pin $cell/CLR];lappend expected $to
            if {$domain eq "mgmt_clk"} {
                bd_paths $timed configuration_recovery $cfg $to $mgmt 10.0
            } else {
                if {$apply} {set_false_path -from $cfg -to $to}
                puts $exceptions "configuration\t$cfg\t$to\t[dict get $::nc_clocks $domain]"
            }
        }
    }
    # Synthesis retains the startup sequencer's own state/enable feedback.
    # These seven management-domain consumers are never reset exceptions.
    set controls [list board_startup/configuration_rst_n_reg/D board_startup/lock_fault_reg/D]
    foreach bit {0 1 2 3 4} {lappend controls [format {board_startup/lock_count_reg[%d]/CE} $bit]}
    foreach to $controls {
        set pin [nb_pin $to];lappend expected $pin
        bd_paths $timed configuration_control $cfg $pin $mgmt 10.0
    }
    if {[lsort [all_fanout -flat -endpoints_only -from [nb_pin board_startup/configuration_rst_n_reg/Q]]] ne [lsort $expected]} {
        error "Raw configuration reset has an unexpected consumer"
    }
    set from [nb_pin {core/reader/startup/source_release_reg[2]/C}]
    foreach chain {ready_meta ready_sync ready_final fault_meta fault_sync fault_final} width {3 3 3 2 2 2} {
        for {set bit 0} {$bit<$width} {incr bit} {
            set to [nb_pin [format {core/reader/startup/%s_reg[%d]/CLR} $chain $bit]]
            set starts [all_fanin -flat -startpoints_only -to $to]
            if {[nc_one $starts reader_reset_source] ne $from} {error "Reader status reset bypasses canonical release: $to $starts"}
            bd_paths $timed reader_status_recovery $from $to [dict get $::nc_clocks reader_clk] 5.0
        }
    }
    set rst [nb_pin clock_generator/inst/mmcme4_adv_inst/RST]
    if {[nc_one [bc_driver $rst] fabric_reset] ne "board_startup/fabric_clock_reset_reg/Q"} {error "Fabric reset source changed"}
    close $timed;close $exceptions
}

proc bd_lock {out apply} {
    set src [nb_pin clock_generator/inst/mmcme4_adv_inst/LOCKED]
    set destinations [list {availability/lock_sync_reg[0]/D} {board_startup/locked_sync_reg[0]/D}]
    set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects $src] -filter {DIRECTION == IN}]
    if {[lsort $loads] ne [lsort $destinations]} {error "LOCKED fanout changed: $loads"}
    set af [open $out/board_lock_arcs.tsv {WRONLY CREAT EXCL}]
    puts $af "phase\tfrom\tto\tdisabled\tuser_disabled"
    set f [open $out/board_lock_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "phase\tfrom\tto\tend_clock\trequirement_ns\tslack_ns"
    foreach phase {before after} {
        if {$phase eq "after" && $apply} {
            foreach to $destinations {set_max_delay -datapath_only 10.000 -from $src -to [nb_pin $to]}
        }
        update_timing
        set arcs [get_timing_arcs -quiet -to $src]
        if {[lsort [get_property FROM_PIN $arcs]] ne [list clock_generator/inst/mmcme4_adv_inst/PWRDWN clock_generator/inst/mmcme4_adv_inst/RST]} {error "LOCKED incoming model arcs differ"}
        foreach arc $arcs {
            set disabled [get_property IS_DISABLED $arc];set user [get_property IS_USER_DISABLED $arc]
            if {$disabled ne "1" || $user ne "0"} {error "LOCKED bound would hide an active model arc"}
            puts $af "$phase\t[get_property FROM_PIN $arc]\t$src\t$disabled\t$user"
        }
        foreach to $destinations {
            if {[nc_one [bc_driver [nb_pin $to]] lock_driver] ne $src} {error "LOCKED input logic changed"}
            set path [nc_one [get_timing_paths -from $src -to [nb_pin $to] -delay_type max -max_paths 1] lock_path]
            if {[get_property ENDPOINT_CLOCK $path] ne "management_refclk"} {error "LOCKED capture clock changed"}
            if {($phase eq "after" || !$apply) && abs([get_property REQUIREMENT $path]-10.0)>0.001} {error "Missing LOCKED route bound"}
            puts $f "$phase\t$src\t$to\t[get_property ENDPOINT_CLOCK $path]\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
        }
    };close $f;close $af
}

proc bd_run {checkpoint out apply} {
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" || [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong board donor"}
    write_xdc -constraints all $out/before.xdc
    bd_census $out
    nc_clock_map $out;nc_constrain $out $apply;nc_topology $out;nc_paths $out
    file rename $out/raw_lock.tsv $out/inherited_raw_lock.tsv
    bc_structure $out;bc_snapshot $out $apply
    wp_connections $out;wp_bounds $out $apply;wp_sync $out
    bd_reset $out $apply;bd_lock $out $apply
    write_xdc -constraints all $out/effective.xdc
    if {$apply} {write_checkpoint $out/board_cdc_bounded.dcp}
    ro_audit $out
    report_cdc -details -file $out/cdc.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_drc -file $out/drc.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f BOARD_CDC_${apply}_EXPORTED_REVIEW_REQUIRED;close $f
    close_design
}
