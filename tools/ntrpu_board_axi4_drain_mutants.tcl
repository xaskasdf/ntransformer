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
    set f [open [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_drain_monitor.sv] r]
    set original [read $f];close $f
    set mutations [list \
        [list stale_tag {captured_tag==req_tag &&} {1'b1 &&} {case=0 tag immediately revokes old ACK}] \
        [list premature_b {(pair_good && pair_id==bid);} {(pair_good && pair_id==bid) || (af && wf && wlast && awid==bid);} {case=7 specific sticky fault}] \
        [list ignore_w_boundary {(pair_ready && !pair_good) || partial_bad ||} {1'b0 ||} {case=9 specific sticky fault}] \
        [list ignore_r_boundary {rvalid && r_available && (rlast!=r_final),} {1'b0,} {case=11 specific sticky fault}]]
    foreach mutation $mutations {
        lassign $mutation name before after expected
        set file [file join $output $name.sv]
        set f [open $file {WRONLY CREAT EXCL}]
        puts -nonewline $f [replace_once $original $before $after];close $f
        create_project drain [file join $output $name] -part xcvu5p-flva2104-2-i
        add_files $file
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_drain_monitor.sv]
        set_property top tb_ntrpu_axi4_drain_monitor [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $output $name drain.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "Fatal: AXI4_ASSERT $expected" $text]<0 || [string first AXI4_DRAIN_SUITE_PASS $text]>=0} {error "Mutant $name did not fail at intended assertion"}
        close_project
        puts "AXI4_MUTANT_REJECTED name=$name"
    }
    puts NTRPU_AXI4_DRAIN_MUTANTS_REJECTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_mutants [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
