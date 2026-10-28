set_param general.maxThreads 2
if {[catch {
    lassign $argv source donor out mode
    if {[llength $argv]!=4 || [file exists $out] || $mode ni {preflight route reopen}} {error "SOURCE DONOR NEW_OUTPUT MODE required"}
    file mkdir $out
    source $source/constrain_hbm_clocktree.tcl
    source $source/insert_hbm_clocktree.tcl
    source $source/audit_hbm_clocktree.tcl
    source $source/audit_hbm_clocktree_topology.tcl
    open_checkpoint $donor
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_clocked"} {error "Wrong checkpoint"}
    if {$mode eq "reopen"} {
        write_xdc -constraints all $out/before.xdc
        file mkdir $out/audit
        ct_topology $out/audit
        ct_audit $out/audit 1
        write_xdc -constraints all $out/after.xdc
    } else {
        write_xdc -constraints all $out/donor.xdc
        ct_insert_buffers $out
        ct_constrain $out
        write_checkpoint $out/buffered.dcp
        if {$mode eq "route"} {
            opt_design
            place_design
            write_checkpoint $out/placed.dcp
            phys_opt_design
            route_design
            write_checkpoint $out/routed.dcp
        }
        file mkdir $out/audit
        ct_topology $out/audit
        ct_audit $out/audit [expr {$mode eq "route"}]
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_CLOCKTREE_${mode}_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
