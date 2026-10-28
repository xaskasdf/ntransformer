set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc replace_once {text before after} {
    set at [string first $before $text]
    if {$at<0 || [string first $before $text [expr {$at+1}]]>=0} {error "Mutation anchor not unique"}
    return [string replace $text $at [expr {$at+[string length $before]-1}] $after]
}
proc axi4_mutants {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set f [open [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_response_guard.sv] r]
    set original [read $f];close $f
    set mutations [list \
        [list reopen_r {!quarantine_r &&} {1'b1 &&} {case=7 cycle=[0-9]+ B retained, late AR cannot authorize orphan R}] \
        [list cut_held_b {assign h_bvalid=cold_rst_n && b_hold;} {assign h_bvalid=cold_rst_n && b_hold && !(|fault_sources);} {case=5 cycle=[0-9]+ healthy held B survives wrong ID}] \
        [list ignore_rlast {&& c_rlast==r_final &&} {&&} {case=9 cycle=[0-9]+ early RLAST blocked, healthy B completes}]]
    foreach mutation $mutations {
        lassign $mutation name before after expected
        set file [file join $output $name.sv]
        set f [open $file {WRONLY CREAT EXCL}]
        puts -nonewline $f [replace_once $original $before $after];close $f
        create_project drain [file join $output $name] -part xcvu5p-flva2104-2-i
        add_files $file
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_response_guard.sv]
        set_property top tb_ntrpu_axi4_response_guard [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $output $name drain.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {![regexp "Fatal: AXI4_GUARD_ASSERT $expected" $text] || [string first AXI4_GUARD_SUITE_PASS $text]>=0} {error "Mutant $name did not fail at intended assertion"}
        close_project
        puts "AXI4_MUTANT_REJECTED name=$name"
    }
    puts NTRPU_AXI4_GUARD_MUTANTS_REJECTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_mutants [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
