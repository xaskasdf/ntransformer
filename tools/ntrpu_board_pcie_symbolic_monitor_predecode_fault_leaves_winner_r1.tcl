set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc fault_leaves_winner_case {output mode vector rows mutant directed input_mutant} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. $vector]]
    create_project predecode_fault_leaves_winner_monitor_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_numeric_pkg_r1.sv fpga/rtl/ntrpu_pcie_rc_update_select_r3.sv fpga/rtl/ntrpu_pcie_request_inventory_r7.sv fpga/rtl/ntrpu_pcie_symbolic_apply_r1.sv fpga/rtl/ntrpu_pcie_symbolic_control_r3.sv fpga/rtl/ntrpu_pcie_symbolic_delivery_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_r2.sv fpga/rtl/ntrpu_pcie_symbolic_effects_r2.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_r1.sv fpga/rtl/ntrpu_pcie_symbolic_header_guard_r1.sv fpga/rtl/ntrpu_pcie_symbolic_late_validation_r1.sv fpga/rtl/ntrpu_pcie_symbolic_monitor_r1.sv fpga/rtl/ntrpu_pcie_symbolic_retirement_r1.sv fpga/tb/ntrpu_pcie_inventory_symbolic_model_r1.sv fpga/tb/ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_late_fault_leaves_winner_r1.sv fpga/rtl/ntrpu_pcie_symbolic_retirement_fault_leaves_winner_r1.sv} {lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]}
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_winner_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_winner_r1_bad_${mutant}.sv]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_winner_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_winner_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"
    if {$directed} {append options " -testplusarg QUIET_DIRECTED"}
    if {$input_mutant ne {}} {append options " -testplusarg $input_mutant"}
    set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project predecode_fault_leaves_winner_monitor_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {} && $input_mutant eq {}} {
        if {$caught || [string first "MONITOR_CANDIDATE_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "MONITOR_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "RETIREMENT_PASS bank=$mode rows=$rows samples=[expr {$rows+5}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Integrated fault-leaf comparison failed: $mode $message"}
    } elseif {[string first "MONITOR_CANDIDATE_PASS" $text]>=0 || ![regexp {Fatal: MONITOR_CANDIDATE_PORT_MISMATCH row=} $text]} {
        error "Integrated fault-leaf control missed intended public comparator: $mode"
    }
    close_project
    puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_CASE_REVIEWED $mode"
}


proc fault_leaves_winner_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    fault_leaves_winner_case $output monitor fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 {} 1 {}
    fault_leaves_winner_case $output directed fpga/tb/data/pcie_symbolic_ced_directed_r1.txt 217 {} 0 {}
    fault_leaves_winner_case $output controls fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 {}
    fault_leaves_winner_case $output range_return fpga/tb/data/pcie_symbolic_retirement_range_r1.txt 277 {} 0 {}
    fault_leaves_winner_case $output rq fpga/tb/data/pcie_rq_admission_integration.txt 8850 {} 0 {}
    fault_leaves_winner_case $output main fpga/tb/data/pcie_request_inventory_r7.txt 22339 {} 0 {}
    fault_leaves_winner_case $output ownership fpga/tb/data/pcie_rc_ownership_integration.txt 46356 {} 0 {}
    fault_leaves_winner_case $output prefix fpga/tb/data/pcie_rc_prefix_integration.txt 544 {} 0 {}
    fault_leaves_winner_case $output lanes fpga/tb/data/pcie_rc_lane_integration_rtl.txt 984 {} 0 {}
    fault_leaves_winner_case $output context fpga/tb/data/pcie_symbolic_value_context.txt 1047 {} 0 {}
    fault_leaves_winner_case $output drained_main fpga/tb/data/pcie_symbolic_retirement_drained_main_r1.txt 22674 {} 0 {}
    fault_leaves_winner_case $output drained_ownership fpga/tb/data/pcie_symbolic_retirement_drained_ownership_r1.txt 65711 {} 0 {}
    fault_leaves_winner_case $output drained_rq fpga/tb/data/pcie_symbolic_retirement_drained_rq_r1.txt 10045 {} 0 {}
    fault_leaves_winner_case $output drained_prefix fpga/tb/data/pcie_symbolic_retirement_drained_prefix_r1.txt 839 {} 0 {}
    fault_leaves_winner_case $output drained_lanes fpga/tb/data/pcie_symbolic_retirement_drained_lanes_r1.txt 1439 {} 0 {}
    fault_leaves_winner_case $output drained_context fpga/tb/data/pcie_symbolic_retirement_drained_context_r1.txt 1132 {} 0 {}
    fault_leaves_winner_case $output omit_rq fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_RQ
    fault_leaves_winner_case $output omit_rc fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_RC
    fault_leaves_winner_case $output omit_cold fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_COLD
    fault_leaves_winner_case $output wrong_sequence fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 WRONG_SEQUENCE
    fault_leaves_winner_case $output wrong_requester fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 WRONG_REQUESTER
    fault_leaves_winner_case $output omit_profile fpga/tb/data/pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0 OMIT_PROFILE
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {fault_leaves_winner_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
