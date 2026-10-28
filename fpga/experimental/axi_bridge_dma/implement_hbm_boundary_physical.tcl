source [file join [file dirname [info script]] implement_hbm_controls_physical.tcl]
source [file join [file dirname [info script]] audit_hbm_boundary_physical.tcl]
rename cp_probe bp_inherited_probe
proc cp_probe {out timed} {
    bp_inherited_probe $out $timed
    bp_probe [file join [file dirname $out] control_boundary] $timed $::bp_candidate
}
