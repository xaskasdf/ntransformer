set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc predecode_case {output mode vector rows mutant directed input_mutant} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. $vector]]
    create_project predecode_monitor_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_numeric_pkg_r1.sv fpga/rtl/ntrpu_pcie_rc_update_select_r3.sv fpga/rtl/ntrpu_pcie_request_inventory_r7.sv fpga/rtl/ntrpu_pcie_symbolic_apply_r1.sv fpga/rtl/ntrpu_pcie_symbolic_control_r3.sv fpga/rtl/ntrpu_pcie_symbolic_delivery_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_r2.sv fpga/rtl/ntrpu_pcie_symbolic_effects_r2.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_r1.sv fpga/rtl/ntrpu_pcie_symbolic_header_guard_r1.sv fpga/rtl/ntrpu_pcie_symbolic_late_validation_r1.sv fpga/rtl/ntrpu_pcie_symbolic_monitor_r1.sv fpga/rtl/ntrpu_pcie_symbolic_retirement_r1.sv fpga/tb/ntrpu_pcie_inventory_symbolic_model_r1.sv fpga/tb/ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_r1.sv} {lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]}
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_monitor_predecode_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_monitor_predecode_r1_bad_${mutant}.sv]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_monitor_predecode_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_monitor_predecode_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"
    if {$directed} {append options " -testplusarg QUIET_DIRECTED"}
    if {$input_mutant ne {}} {append options " -testplusarg $input_mutant"}
    set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_MONITOR_PREDECODE_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project predecode_monitor_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {} && $input_mutant eq {}} {
        if {$caught || [string first "MONITOR_CANDIDATE_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "MONITOR_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "RETIREMENT_PASS bank=$mode rows=$rows samples=[expr {$rows+5}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Predecode comparison failed: $mode $message"}
    } elseif {[string first "MONITOR_CANDIDATE_PASS" $text]>=0 || ![regexp {Fatal: MONITOR_CANDIDATE_PORT_MISMATCH row=} $text]} {
        error "Predecode control missed intended public comparator: $mode"
    }
    close_project
    puts "NTRPU_MONITOR_PREDECODE_CASE_REVIEWED $mode"
}

proc predecode_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    predecode_case $output monitor fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 {} 1 {}
    predecode_case $output directed fpga/tb/data/pcie_symbolic_ced_directed_r1.txt 217 {} 0 {}
    predecode_case $output controls fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 {}
    predecode_case $output range_return fpga/tb/data/pcie_symbolic_retirement_range_r1.txt 277 {} 0 {}
    predecode_case $output rq fpga/tb/data/pcie_rq_admission_integration.txt 8850 {} 0 {}
    predecode_case $output bad_omit_c fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_c 1 {}
    predecode_case $output bad_omit_e fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_e 1 {}
    predecode_case $output bad_omit_d fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_d 1 {}
    predecode_case $output bad_omit_a fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_a 1 {}
    predecode_case $output bad_omit_b fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_b 1 {}
    predecode_case $output bad_extra_retired fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 extra_retired 1 {}
    predecode_case $output bad_omit_early_fault fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_early_fault 1 {}
    predecode_case $output bad_omit_late_fault fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_late_fault 1 {}
    predecode_case $output bad_omit_cold fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_cold 1 {}
    predecode_case $output bad_omit_observing fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_observing 1 {}
    predecode_case $output bad_omit_requester fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_requester 1 {}
    predecode_case $output bad_omit_attributes fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_attributes 1 {}
    predecode_case $output bad_omit_rq_ready fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_rq_ready 1 {}
    predecode_case $output bad_omit_rc_ready fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_rc_ready 1 {}
    predecode_case $output bad_omit_raw_rq fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_raw_rq 1 {}
    predecode_case $output bad_omit_raw_rc fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 omit_raw_rc 1 {}
    predecode_case $output bad_drop_admission_zero fpga/tb/data/pcie_rq_admission_integration.txt 8850 drop_admission_zero 0 {}
    predecode_case $output bad_drop_admission_one fpga/tb/data/pcie_rq_admission_integration.txt 8850 drop_admission_one 0 {}
    predecode_case $output bad_swap_admission_selects fpga/tb/data/pcie_rq_admission_integration.txt 8850 swap_admission_selects 0 {}
    predecode_case $output bad_rotate_admission_rows fpga/tb/data/pcie_rq_admission_integration.txt 8850 rotate_admission_rows 0 {}
    predecode_case $output bad_delayed_admission_selects fpga/tb/data/pcie_rq_admission_integration.txt 8850 delayed_admission_selects 0 {}
    predecode_case $output omit_rq fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_RQ
    predecode_case $output omit_rc fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_RC
    predecode_case $output omit_cold fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_COLD
    predecode_case $output wrong_sequence fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 WRONG_SEQUENCE
    predecode_case $output wrong_requester fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 WRONG_REQUESTER
    predecode_case $output omit_profile fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_PROFILE
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_GATE_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {predecode_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
