# Recover the completed V3 synthesis; verify native alias in the loaded V2 reference.
set_param general.maxThreads 2
source [file join [file dirname [info script]] audit_guard_v3.tcl]
if {[catch {
    if {[llength $argv]!=3} {error "SYNTH_DCP NEW_OUTPUT REFERENCE_DCP required"}
    lassign $argv dcp out reference
    if {[file exists $out]} {error "New output required"};file mkdir $out
    v3_reference_alias $reference $out/reference_alias.tsv
    open_checkpoint $dcp
    set g guarded/guard
    foreach {from to bound} [list request_reg/C {echo_a_reg[0]/D} 4.000 {echo_a_reg[2]/C} {echo_m_reg[0]/D} 10.000 armed_reg/C {armed_a_reg[0]/D} 4.000] {
        set_max_delay -datapath_only $bound -from [v3_pin $g/$from] -to [v3_pin $g/$to]
    }
    for {set n 0} {$n<3} {incr n} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [v3_pin [format {%s/availability[%d].release_m_reg[2]/C} $g $n]] -to [v3_pin [format {%s/availability[%d].release_a_reg[%d]/CLR} $g $n $i]]
        }
    }
    foreach {n reg} {0 user_lnk_up_reg 1 user_reset_reg} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [v3_pin ntrpu_bridge_i/inst/pcie4c_ip_i/inst/$reg/C] -to [v3_pin [format {%s/availability[%d].release_m_reg[%d]/CLR} $g $n $i]]
        }
    }
    for {set i 0} {$i<3} {incr i} {set_false_path -from [v3_pin $g/fault_reg/C] -to [v3_pin [format {%s/stop_a_reg[%d]/PRE} $g $i]]}
    opt_design
    write_checkpoint $out/optimized.dcp
    v3_evidence $out/optimized
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f V3_SHARED_NATIVE_RECOVERY_EXPORTED;close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
