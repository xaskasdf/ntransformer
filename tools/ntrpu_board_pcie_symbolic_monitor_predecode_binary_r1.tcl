set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_monitor_predecode_binary {seed output} {
    global ntrpu_script_dir
    if {[file exists $output] || ![file isfile $seed]} {error "Invalid seed/output"}
    file mkdir $output
    file copy $seed [file join $output input_funcsim.v]
    set_param general.maxThreads 2
    create_project symbolic_monitor_predecode_binary [file join $output project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    lappend sources [file join $output input_funcsim.v]
    foreach name {ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv ntrpu_pcie_symbolic_late_validation_r1.sv ntrpu_pcie_symbolic_monitor_r1.sv ntrpu_pcie_symbolic_retirement_r1.sv ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga tb monitor_mapped_refs_r1 $name]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_monitor_predecode_binary_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_monitor_predecode_binary_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set cases {
        {monitor pcie_symbolic_monitor_directed_r2.txt 252 {} 1}
        {controls pcie_symbolic_numeric_ab_controls_r1.txt 352 {} 0}
        {omit_rq pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_RQ 0}
        {omit_rc pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_RC 0}
        {omit_cold pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_COLD 0}
        {wrong_sequence pcie_symbolic_numeric_ab_controls_r1.txt 352 WRONG_SEQUENCE 0}
        {wrong_requester pcie_symbolic_numeric_ab_controls_r1.txt 352 WRONG_REQUESTER 0}
        {omit_profile pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_PROFILE 0}
        {range_return pcie_symbolic_retirement_range_r1.txt 277 {} 0}
        {main pcie_symbolic_ced_binary_raw_main_r1.txt 18538 {} 0}
        {ownership pcie_rc_ownership_integration.txt 46356 {} 0}
        {rq pcie_rq_admission_integration.txt 8850 {} 0}
        {prefix pcie_rc_prefix_integration.txt 544 {} 0}
        {lanes pcie_rc_lane_integration_binary.txt 632 {} 0}
        {context pcie_symbolic_value_context_binary.txt 997 {} 0}
        {drained_main pcie_symbolic_retirement_binary_drained_main_r1.txt 18858 {} 0}
        {drained_ownership pcie_symbolic_retirement_drained_ownership_r1.txt 65711 {} 0}
        {drained_rq pcie_symbolic_retirement_drained_rq_r1.txt 10045 {} 0}
        {drained_prefix pcie_symbolic_retirement_drained_prefix_r1.txt 839 {} 0}
        {drained_lanes pcie_symbolic_retirement_binary_drained_lanes_r1.txt 932 {} 0}
        {drained_context pcie_symbolic_retirement_binary_drained_context_r1.txt 1062 {} 0}
        {directed pcie_symbolic_ced_binary_directed_r1.txt 191 {} 0}
    }
    foreach item $cases {
        lassign $item bank vector rows mutation targeted
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$bank -testplusarg ROWS=$rows"
        if {$mutation ne {}} {append options " -testplusarg $mutation"}
        if {$targeted} {append options " -testplusarg QUIET_DIRECTED"}
        set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
        puts "NTRPU_MONITOR_PREDECODE_MAPPED_CASE_START $bank"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set sim [file join $output project symbolic_monitor_predecode_binary.sim sim_1 behav xsim]
        set folder [file join $output $bank];file mkdir $folder
        foreach file [glob -nocomplain -directory $sim *.log *.prj *.sh *.tcl] {file copy $file $folder}
        set f [open [file join $folder simulate.log] r];set text [read $f];close $f
        if {$mutation eq {}} {
            if {$caught || [string first "MONITOR_MAPPED_PASS bank=$bank rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] probe_samples=[expr {$targeted ? 20 : 0}] mode=0 ports=447" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Mapped monitor mismatch: $bank $message"}
        } else {
            if {[string first "MONITOR_MAPPED_PASS" $text]>=0 || [string first "Fatal: MONITOR_MAPPED_PORT_MISMATCH row=" $text]<0} {
                error "Mapped boundary control not rejected: $bank"
            }
        }
        if {$bank eq "monitor" && [string first "probes=10" $text]<0} {error "Missing binary quiet probes"}
        puts "NTRPU_MONITOR_PREDECODE_MAPPED_CASE_REVIEWED $bank"
    }
    close_project
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_MAPPED_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_monitor_predecode_binary [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
