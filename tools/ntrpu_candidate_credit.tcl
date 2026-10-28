set root [file normalize [file join [file dirname [info script]] ..]]
proc credit_test {root output} {
    if {[file exists $output]} {error "Expected new directory"}
    file mkdir $output
    set_param general.maxThreads 2
    exec python3 -B [file join $root tools prepare_ntrpu_candidate_dispatch.py] [file join $output generated]
    create_project credit [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $output generated ntrpu_destination_candidate.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_candidate_credit.sv]
    set_property top tb_ntrpu_candidate_credit [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output project credit.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output credit.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "CU SUITE_PASS cases=2" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {
        error "Credit withdrawal test failed"
    }
    puts NTRPU_CANDIDATE_CREDIT_COMPLETE
    close_project
}
if {[llength $argv]!=1} {error "Expected output directory"}
if {[catch {credit_test $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
