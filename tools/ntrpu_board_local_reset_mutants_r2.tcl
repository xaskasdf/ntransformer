# Negative RTL controls. Each project compiles one private mutated copy of the
# sealed actuator with the unchanged positive testbench. Never edit that source.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_local_reset_mutants {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set original [file join $ntrpu_script_dir .. fpga rtl ntrpu_local_reset_actuator.sv]
    set f [open $original r];set source [read $f];close $f
    set changes [list \
        [list ignore_generation {saved_generation==req_generation && } {} {case=6 generation replacement masks old ACK}] \
        [list early_release {!command_started && state==HELD && peers_held} {!command_started && (state==HELD || state==COUNT) && peers_held} {case=2 early ready cannot release before counters}]]
    foreach mutation $changes {
        lassign $mutation name before after expected
        set first [string first $before $source]
        if {$first<0 || [string first $before $source [expr {$first+1}]]>=0} {error "Mutation anchor is not unique: $name"}
        set mutant [string replace $source $first [expr {$first+[string length $before]-1}] $after]
        set private [file join $output ${name}.sv]
        set f [open $private {WRONLY CREAT EXCL}];puts -nonewline $f $mutant;close $f
        set project [file join $output $name]
        create_project $name $project -part xcvu5p-flva2104-2-i
        set_property XPM_LIBRARIES {XPM_CDC} [current_project]
        add_files $private
        add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_reset_supervisor.sv]
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_local_reset_actuator_r2.sv]
        set_property top tb_ntrpu_local_reset_actuator [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        set rc [catch {run all} detail]
        puts "LOCAL_MUTANT_RUN name=$name catch=$rc detail=$detail"
        close_sim
        set log [file join $project ${name}.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "Fatal: LOCAL_ASSERT $expected" $text]<0 || [string first LOCAL_SUITE_PASS $text]>=0} {error "Expected specific early assertion did not reject $name"}
        close_project
        puts "LOCAL_MUTANT_REJECTED name=$name expected=$expected"
    }
    puts NTRPU_LOCAL_RESET_MUTANTS_REJECTED
}
if {[catch {ntrpu_local_reset_mutants $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
