# Reopen the saved synthesis and audit references; never change constraints.
set_param general.maxThreads 2
if {[catch {
    lassign $argv checkpoint out
    if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" ||
        [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled"} {error "Wrong synthesized donor"}
    write_xdc -constraints all $out/before.xdc
    source [file join [file dirname [info script]] audit_hbm_reference_ownership.tcl]
    ro_audit $out
    write_xdc -constraints all $out/after.xdc
    set f [open $out/ports.tsv {WRONLY CREAT EXCL}];puts $f "port\tdirection"
    foreach p [lsort [get_ports]] {puts $f "$p\t[get_property DIRECTION [get_ports $p]]"};close $f
    set f [open $out/native_instances.tsv {WRONLY CREAT EXCL}];puts $f "cell\treference"
    foreach name {ntrpu_bridge_i hbm_binding/memory clock_generator core} {
        set cell [ro_one [get_cells -quiet $name] $name]
        puts $f "$name\t[get_property REF_NAME $cell]"
    };close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_REFERENCE_REOPENED_EXPORTED
    close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
