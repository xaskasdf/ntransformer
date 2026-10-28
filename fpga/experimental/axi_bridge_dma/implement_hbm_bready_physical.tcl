source [file join [file dirname [info script]] implement_hbm_boundary_physical.tcl]
source [file join [file dirname [info script]] audit_hbm_bready_physical.tcl]
rename cp_probe wb_inherited_probe
proc cp_probe {out timed} {
    wb_inherited_probe $out $timed
    wb_probe [file join [file dirname $out] registered_bready] $timed
}
