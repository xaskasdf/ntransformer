if {[catch {
    if {[llength $argv]!=2} {error "CHECKPOINT NEW_OUTPUT required"}
    source [file join [file dirname [info script]] audit_hbm_board_cdc.tcl]
    bd_run {*}$argv 0
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
