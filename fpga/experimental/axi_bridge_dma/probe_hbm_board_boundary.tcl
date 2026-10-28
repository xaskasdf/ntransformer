# Read-only inventory of the registered-readiness native checkpoint.
set_param general.maxThreads 2
proc bb_one {objects why} {
    if {[llength $objects] != 1} {error "$why: expected one object, got $objects"}
    return [lindex $objects 0]
}
proc bb_text {s} {return [string map [list "\t" " " "\n" " " "\r" " "] $s]}
proc bb_props {path objects properties} {
    set f [open $path {WRONLY CREAT EXCL}];puts $f "object\tproperty\tvalue"
    foreach o [lsort $objects] {
        set supported [list_property $o]
        foreach p $properties {
            if {[lsearch -exact $supported $p]>=0} {puts $f "$o\t$p\t[bb_text [get_property $p $o]]"}
        }
    };close $f
}
if {[catch {
    if {[llength $argv]!=2} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv checkpoint out
    if {[file exists $out]} {error "Fresh output required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled"} {error "Unexpected native composition"}
    write_xdc -constraints all $out/before.xdc
    bb_props $out/package.tsv [get_package_pins] {PIN_FUNC BANK IS_BONDED IS_GLOBAL_CLK IS_MASTER DIFF_PAIR_PIN IS_USED PIN_RESTRICTIONS}
    bb_props $out/banks.tsv [get_iobanks] {BANK_TYPE VCCO INTERNAL_VREF DCI_CASCADE}
    set f [open $out/gc_sites.tsv {WRONLY CREAT EXCL}]
    puts $f "ball\tfunction\tbank\tsite\tsite_type\tclock_region\tslr"
    foreach ball [lsort [get_package_pins -filter {IS_GLOBAL_CLK == 1 && IS_BONDED == 1}]] {
        foreach site [get_sites -of_objects $ball] {
            puts $f [join [list $ball [get_property PIN_FUNC $ball] [get_property BANK $ball] $site [get_property SITE_TYPE $site] [join [get_clock_regions -of_objects $site] ,] [join [get_slrs -of_objects $site] ,]] "\t"]
        }
    };close $f
    set f [open $out/ports.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tpackage_pin\tiostandard\tleaf_pin\tref\tloc"
    foreach port [lsort [get_ports]] {
        set leaves [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $port]]
        if {![llength $leaves]} {error "Unconnected external port $port"}
        foreach pin [lsort $leaves] {
            set cell [bb_one [get_cells -of_objects $pin] leaf_cell]
            puts $f [join [list $port [get_property DIRECTION $port] [get_property PACKAGE_PIN $port] [get_property IOSTANDARD $port] $pin [get_property REF_NAME $cell] [get_property LOC $cell]] "\t"]
        }
    };close $f
    bb_props $out/resources.tsv [get_cells -hier -filter {IS_PRIMITIVE == 1 && (REF_NAME =~ HBM* || REF_NAME =~ MMCME* || REF_NAME =~ PCIE4* || REF_NAME =~ GTYE4* || REF_NAME =~ IBUF* || REF_NAME =~ BUFG*)}] {REF_NAME LOC BEL IS_LOC_FIXED IS_BEL_FIXED}
    set f [open $out/resource_sites.tsv {WRONLY CREAT EXCL}]
    puts $f "site\ttype\tclock_region\tslr"
    foreach site [lsort [get_sites -filter {SITE_TYPE =~ HBM* || SITE_TYPE =~ MMCME* || SITE_TYPE =~ IBUFDS_GTE4 || SITE_TYPE =~ PCIE4*}]] {
        puts $f [join [list $site [get_property SITE_TYPE $site] [join [get_clock_regions -of_objects $site] ,] [join [get_slrs -of_objects $site] ,]] "\t"]
    };close $f
    set f [open $out/clock_roots.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tleaf_pin\tref\tclock\tperiod_ns"
    foreach port {sys_clk_p fabric_ref_clk_in mgmt_clk_in hbm_ref0_clk_in hbm_ref1_clk_in} {
        set clock [bb_one [get_clocks -of_objects [get_ports $port]] reference_clock]
        foreach pin [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_ports $port]]] {
            puts $f "$port\t$pin\t[get_property REF_NAME [get_cells -of_objects $pin]]\t$clock\t[get_property PERIOD $clock]"
        }
    };close $f
    set f [open $out/readiness_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\ttype\trequirement_ns\tslack_ns"
    foreach from {availability/fault_reg/C {availability/ready_mgmt_reg[0]/C} {availability/ready_mgmt_reg[1]/C}} {
        foreach type {max min} {
            set p [bb_one [get_timing_paths -quiet -from [get_pins $from] -to [get_pins loader_readiness/ready_reg/D] -delay_type $type -max_paths 1] readiness_path]
            puts $f "$from\tloader_readiness/ready_reg/D\t$type\t[get_property REQUIREMENT $p]\t[get_property SLACK $p]"
        }
    };close $f
    check_timing -verbose -file $out/check_timing.rpt
    report_drc -file $out/drc.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/full_cdc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_timing -delay_type min -max_paths 20 -file $out/worst_hold.rpt
    report_timing -delay_type max -max_paths 20 -file $out/worst_setup.rpt
    report_clocks -file $out/clocks.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_BOARD_BOUNDARY_EXPORTED_NO_ASSIGNMENTS;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
