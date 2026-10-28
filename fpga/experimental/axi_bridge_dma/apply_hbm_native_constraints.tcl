source [file join [file dirname [info script]] audit_hbm_native_constraints.tcl]
if {[catch {
    if {[llength $argv]!=2} {error "DONOR NEW_OUTPUT required"}
    nc_run [lindex $argv 0] [lindex $argv 1] apply
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
