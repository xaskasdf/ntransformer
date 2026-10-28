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
    set f [open [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_retained_admission.sv] r]
    set original [read $f];close $f
    set mutations [list \
        [list partial_gap {+(w_partial!=0 || up_wvalid);} {+up_wvalid;} {case=10 cycle=[0-9]+ partial W gap reserves missing AW and WLAST}] \
        [list cut_held_aw {wire permit_aw=held_aw ||} {wire permit_aw=1'b0 ||} {case=6 cycle=[0-9]+ held AW survives capacity loss}] \
        [list cut_held_w {wire permit_w=held_w ||} {wire permit_w=1'b0 ||} {case=11 cycle=[0-9]+ held W survives capacity loss}] \
        [list early_local_last {state==REJECT?(local_rleft==1):g_rlast} {state==REJECT?(local_rleft<=2):g_rlast} {reviewer_only}]]
    foreach mutation $mutations {
        lassign $mutation name before after expected
        set file [file join $output $name.sv]
        set f [open $file {WRONLY CREAT EXCL}]
        puts -nonewline $f [replace_once $original $before $after];close $f
        create_project drain [file join $output $name] -part xcvu5p-flva2104-2-i
        add_files $file
        add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_response_guard.sv]
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_retained_admission_r2.sv]
        set_property top tb_ntrpu_axi4_retained_admission_r2 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $output $name drain.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {$expected eq "reviewer_only"} {
            if {[string first "AXI4_ADMISSION_SUITE_PASS cases=13" $text]<0 || [regexp {(^|\n)[ \t]*Fatal:} $text]} {error "Reviewer-only mutant did not finish the HDL bank"}
        } elseif {![regexp "Fatal: AXI4_ADMISSION_ASSERT $expected" $text] || [string first AXI4_ADMISSION_SUITE_PASS $text]>=0} {error "Mutant $name did not fail at intended HDL assertion"}
        close_project
        if {$expected eq "reviewer_only"} {puts "AXI4_MUTANT_AWAITING_REVIEW name=$name"} else {puts "AXI4_MUTANT_REJECTED name=$name"}
    }
    puts NTRPU_AXI4_ADMISSION_MUTANTS_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_mutants [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
