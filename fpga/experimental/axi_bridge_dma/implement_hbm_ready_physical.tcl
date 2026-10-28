# Extend the frozen complete-board flow with the new READY checks only.
source [file join [file dirname [info script]] implement_hbm_capture_physical.tcl]
source [file join [file dirname [info script]] audit_hbm_ready_physical.tcl]
rename cp_probe rp_inherited_capture_probe
proc cp_probe {out timed} {
    rp_inherited_capture_probe $out $timed
    rp_probe [file join [file dirname $out] registered_ready] $timed
}
