set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc admission_count_case {output mode candidate} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project admission_count_focus [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_rc_update_select_commit_priority_named_r1.sv fpga/rtl/ntrpu_pcie_symbolic_control_predecode_rq_bitmap_r1.sv} {
        lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]
    }
    lappend sources [file normalize [file join $ntrpu_script_dir .. $candidate]]
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_control_predecode_admission_count_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_control_predecode_admission_count_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rq_admission_integration.txt]]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_ADMISSION_COUNT_FOCUS_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project admission_count_focus.sim sim_1 behav xsim simulate.log]
    set stream [open $logfile r]; set text [read $stream]; close $stream
    file copy $logfile [file join $folder simulate.log]
    if {$mode eq "positive"} {
        if {$caught || [string first "ADMISSION_COUNT_PASS rows=51110 pre=51110 post=51110 control_cases=8192 kind_cases=256 handshake_cases=64 vector_rows=8850 record_bits=3574 extended_kind_cases=1024 extended_handshake_cases=1024 count_bits=13" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RQ bitmap positive failed: $message"}
    } elseif {[string first "ADMISSION_COUNT_PASS" $text]>=0 || ![regexp {Fatal: ADMISSION_COUNT_COUNTER_MISMATCH row=} $text]} {
        error "RQ bitmap mutant missed intended comparator: $mode"
    }
    close_project
    puts "NTRPU_ADMISSION_COUNT_FOCUS_CASE_REVIEWED $mode"
}
proc admission_count_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    admission_count_case $output positive fpga/rtl/ntrpu_pcie_symbolic_control_predecode_admission_count_r1.sv
    foreach mutation {drop_lane_three collapse_two count_blocked add_completions} {
        admission_count_case $output $mutation fpga/tb/ntrpu_pcie_symbolic_control_predecode_admission_count_bad_${mutation}_r1.sv
    }
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_FOCUS_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {admission_count_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
