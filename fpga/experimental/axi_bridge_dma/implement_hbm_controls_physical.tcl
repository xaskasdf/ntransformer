# Extend the unchanged READY/capture/reset flow with read-only control probes.
source [file join [file dirname [info script]] implement_hbm_ready_physical.tcl]
source [file join [file dirname [info script]] audit_hbm_controls_physical.tcl]
rename cp_probe pp_inherited_probe
proc cp_probe {out timed} {
    pp_inherited_probe $out $timed
    pp_probe [file join [file dirname $out] control_predictors] $timed $::pp_candidate
}
