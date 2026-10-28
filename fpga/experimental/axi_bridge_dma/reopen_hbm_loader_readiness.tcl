source [file join [file dirname [info script]] audit_hbm_loader_readiness.tcl]
if {[catch {
    if {[llength $argv]!=2} {error "DONOR NEW_OUTPUT required"}
    wp_run [lindex $argv 0] [lindex $argv 1] 0
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
