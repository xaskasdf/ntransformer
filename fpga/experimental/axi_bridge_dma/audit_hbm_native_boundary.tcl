# Add only the raw LOCKED route budget; inventory the still-open boundary.
source [file join [file dirname [info script]] audit_hbm_native_constraints.tcl]
proc nb_pin {name} {
    set p [nc_one [get_pins -quiet $name] $name]
    if {[get_property NAME $p] ne $name} {error "Non-exact boundary pin $name"}
    return $p
}
proc nb_lock {out apply} {
    set src [nb_pin clock_generator/inst/mmcme4_adv_inst/LOCKED]
    set dst [nb_pin {availability/lock_sync_reg[0]/D}]
    if {[get_property REF_NAME [get_cells -of_objects $src]] ne "MMCME4_ADV"} {error "LOCKED primitive changed"}
    set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $dst] -filter {DIRECTION == OUT}]
    set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects $src] -filter {DIRECTION == IN}]
    if {$drivers ne $src || $loads ne $dst} {error "Raw LOCKED internal fanout changed"}
    set arcs [get_timing_arcs -quiet -to $src]
    set f [open $out/lock_source_arcs.tsv {WRONLY CREAT EXCL}];puts $f "from\tto"
    foreach arc $arcs {puts $f "[get_property FROM_PIN $arc]\t[get_property TO_PIN $arc]"};close $f
    # The primitive exposes RST/PWRDWN arcs even though the unconstrained
    # data path starts at LOCKED. Preserve their full properties across the
    # constraint. Keep the exact segmentation warning visible for the separate
    # reviewer, which must prove these arcs were already disabled by the model.
    set froms [lsort [get_property FROM_PIN $arcs]]
    if {$froms ne [list clock_generator/inst/mmcme4_adv_inst/PWRDWN clock_generator/inst/mmcme4_adv_inst/RST]} {error "Unexpected incoming LOCKED arcs"}
    set af [open $out/lock_arc_properties.tsv {WRONLY CREAT EXCL}]
    puts $af "phase\tfrom\tto\tproperty\tvalue"
    set f [open $out/lock_path.tsv {WRONLY CREAT EXCL}]
    puts $f "phase\tfrom\tto\tstart_clock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach phase {before after} {
        if {$phase eq "after" && $apply} {set_max_delay -datapath_only 10.000 -from $src -to $dst}
        update_timing
        foreach arc [get_timing_arcs -quiet -to $src] {
            foreach property [lsort [list_property $arc]] {
                set value [string map [list "\n" {\n} "\t" {\t}] [get_property $property $arc]]
                puts $af "$phase\t[get_property FROM_PIN $arc]\t[get_property TO_PIN $arc]\t$property\t$value"
            }
        }
        set path [nc_one [get_timing_paths -from $src -to $dst -delay_type max -max_paths 1] "raw LOCKED path"]
        if {[get_property STARTPOINT_PIN $path] ne $src || [get_property ENDPOINT_PIN $path] ne $dst} {error "LOCKED path identity changed"}
        if {[get_property ENDPOINT_CLOCK $path] ne "management_refclk"} {error "LOCKED capture clock changed"}
        if {($phase eq "after" || !$apply) && abs([get_property REQUIREMENT $path]-10.0)>0.001} {error "LOCKED max delay missing/overridden"}
        puts $f [join [list $phase [get_property STARTPOINT_PIN $path] [get_property ENDPOINT_PIN $path] [get_property STARTPOINT_CLOCK $path] [get_property ENDPOINT_CLOCK $path] [get_property REQUIREMENT $path] [get_property SLACK $path] [get_property DATAPATH_DELAY $path]] "\t"]
        report_timing -from $src -to $dst -delay_type max -max_paths 1 -file $out/lock_${phase}.rpt
    };close $f;close $af
}
proc nb_ports {out} {
    set f [open $out/boundary_ports.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tpackage_pin\tiostandard\tport_clocks\tpeer_count\tpeer_clocks\tpeer_sample"
    foreach port [lsort [get_ports]] {
        set p [get_ports $port];set dir [get_property DIRECTION $p]
        if {$dir eq "IN"} {
            set peers [all_fanout -flat -endpoints_only -from $p]
        } elseif {$dir eq "OUT"} {
            set peers [all_fanin -flat -startpoints_only -to $p]
        } else {error "Unexpected inout port $port"}
        set cells [get_cells -quiet -of_objects $peers]
        set clocks [lsort -unique [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $cells -filter {IS_CLOCK == 1}]]]
        puts $f [join [list $port $dir [get_property PACKAGE_PIN $p] [get_property IOSTANDARD $p] [join [get_clocks -quiet -of_objects $p] |] [llength $peers] [join $clocks |] [join [lrange [lsort $peers] 0 7] |]] "\t"]
    };close $f
}
proc nb_run {checkpoint out mode} {
    if {[file exists $out] || $mode ni {apply reopen}} {error "New output and apply/reopen required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_native_join"} {error "Wrong native donor"}
    write_xdc -constraints all $out/before.xdc
    nc_clock_map $out
    nc_constrain $out 0
    nb_lock $out [expr {$mode eq "apply"}]
    nc_topology $out
    # This is the inherited stage's structural observation, not this stage's
    # constraint status. lock_path.tsv supplies the new measured requirement.
    file rename $out/raw_lock.tsv $out/inherited_raw_lock.tsv
    nc_paths $out
    nb_ports $out
    write_xdc -constraints all $out/after.xdc
    report_cdc -details -file $out/cdc.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    check_timing -verbose -file $out/check_timing.rpt
    if {$mode eq "apply"} {write_checkpoint $out/bounded.dcp}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f NATIVE_BOUNDARY_${mode}_EXPORTED_PENDING_REVIEW;close $f
    close_design
}
