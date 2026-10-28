set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc commit_priority_mask_rows_case {output mode mutant top vector_arg} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project rc_commit_priority_mask_rows_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    if {$mutant eq {}} {
        set source [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_rc_update_select_commit_priority_mask_rows_r1.sv]
    } else {
        set source [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_rc_update_select_commit_priority_mask_rows_bad_${mutant}_r1.sv]
    }
    add_files $source
    set tests {}
    foreach name {ntrpu_pcie_rc_update_select_reference_r1.sv tb_ntrpu_pcie_rc_event_ancestry.sv tb_ntrpu_pcie_rc_update_select_r1.sv} {
        lappend tests [file normalize [file join $ntrpu_script_dir .. fpga tb $name]]
    }
    add_files -fileset sim_1 $tests
    set_property top $top [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options $vector_arg] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_COMMIT_PRIORITY_MASK_ROWS_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project rc_commit_priority_mask_rows_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant ne {}} {
        if {![regexp {(?m)^Fatal: RC_OWNERSHIP_MISMATCH check=[0-9]+ actual=[0-9a-fA-F]+ expected=[0-9a-fA-F]+ flags=} $text] ||
            [string first RC_OWNERSHIP_PASS $text]>=0} {error "Expected source mutation not detected: $mode"}
    } elseif {$mode eq "ancestry"} {
        if {$caught || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text] ||
            ![regexp {(?m)^RC_OWNERSHIP_PASS rows=155168 checks=1153296 unknown_checks=994288$} $text] ||
            ![regexp {(?m)^RC_EVENT_ANCESTRY_PASS checks=32768 unknown_checks=32256$} $text]} {
            error "Incomplete event/ownership comparison: $message"
        }
    } else {
        if {$caught || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text] ||
            ![regexp {(?m)^RC_UPDATE_ANCESTRY_PASS rows=89632$} $text]} {error "Incomplete selection comparison: $message"}
    }
    close_project
    puts "NTRPU_COMMIT_PRIORITY_MASK_ROWS_CASE_REVIEWED $mode"
}
proc commit_priority_mask_rows {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set ownership [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_ownership.txt]]
    set selection [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_update_ancestry.txt]]
    commit_priority_mask_rows_case $output ancestry {} tb_ntrpu_pcie_rc_event_ancestry "-testplusarg OWNERSHIP_VECTORS=$ownership"
    commit_priority_mask_rows_case $output selection {} tb_ntrpu_pcie_rc_update_select_r1 "-testplusarg SELECT_VECTORS=$selection"
    foreach mutant {omit_origin_zero reverse_origin omit_header_three} {
        commit_priority_mask_rows_case $output $mutant $mutant tb_ntrpu_pcie_rc_event_ancestry "-testplusarg OWNERSHIP_VECTORS=$ownership"
    }
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_MASK_ROWS_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {commit_priority_mask_rows [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
