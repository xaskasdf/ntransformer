# Reuse the exact established physical flow. Only fixture generation and the
# additional routed comparison change; all constraints and reports are shared.
set base [file join [file dirname [info script]] ntrpu_transport_study.tcl]
set f [open $base r];set script [read $f];close $f
set script [string map {prepare_ntrpu_transport_study.py prepare_ntrpu_overlap_physical.py} $script]
set first {        add_files -fileset sim_1 [file join $generated ${top}.sv]}
set last {        close_project}
set begin [string first $first $script]
set end [string first $last $script $begin]
if {$begin<0 || $end<$begin} {error "Reference simulation flow changed"}
set replacement {
        foreach {top filename} {
            tb_ntrpu_transport_netlist simulation.log
            tb_ntrpu_overlap_netlist_extra extra.log
        } {
            add_files -fileset sim_1 [file join $generated ${top}.sv]
            set_property top $top [get_filesets sim_1]
            set_property xsim.elaborate.debug_level off [get_filesets sim_1]
            set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project transport_study.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output $filename]
            set f [open $log r];set text [read $f];close $f
            if {[string first {TN ALL_PASS checks=} $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Routed comparison failed: $filename"}
        }
        close_project}
set script [string replace $script $begin [expr {$end+[string length $last]-1}] $replacement]
if {[lindex $argv 1] ni {synth place route netlist}} {error "Physical stages only"}
eval $script
