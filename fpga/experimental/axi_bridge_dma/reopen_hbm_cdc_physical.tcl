set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=3} {error "SOURCE ROUTED_DCP NEW_OUTPUT required"}
    lassign $argv source dcp out
    if {[file exists $out]} {error "New output required"};file mkdir $out
    open_checkpoint $dcp
    source $source/constrain_hbm_cdc_physical.tcl
    source $source/audit_hbm_cdc_physical.tcl
    source $source/audit_hbm_cdc_topology.tcl
    write_xdc -constraints all $out/before.xdc
    hc_audit $out/audit
    hc_topology $out/audit
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CDC_REOPENED_READONLY_AUDIT_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
