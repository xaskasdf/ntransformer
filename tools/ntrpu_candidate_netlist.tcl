# Functional simulation of unchanged full routed netlist; no SDF or implementation.
set root [file normalize [file join [file dirname [info script]] ..]]
proc run_netlist {root output netlist} {
    if {[file exists $output]} {error "Expected fresh output"}
    file mkdir $output
    set_param general.maxThreads 2
    set generated [file join $output generated]
    exec python3 -B [file join $root tools prepare_ntrpu_candidate_netlist.py] $generated --netlist $netlist
    create_project candidate_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    foreach file [glob [file join $generated ref_*.sv]] {add_files $file}
    add_files $netlist
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    foreach {top marker name} {
        tb_ntrpu_candidate_netlist_dispatch {RS SUITE_PASS cases=8 } simulation
        tb_ntrpu_candidate_netlist_extra {CP EXTRA_PASS cases=2 } extra
    } {
        add_files -fileset sim_1 [file join $generated ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project candidate_netlist.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first $marker $text]<0 || [string first {CN ALL_PASS checks=} $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Netlist $name failed"}
    }
    close_project
    puts NTRPU_CANDIDATE_NETLIST_COMPLETE
}
if {[llength $argv]!=2} {error "Expected output and netlist"}
if {[catch {run_netlist $root [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
