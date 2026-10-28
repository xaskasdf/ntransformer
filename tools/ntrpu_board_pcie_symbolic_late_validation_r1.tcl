set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc late_case {output mode vector rows mutant} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
    create_project symbolic_late_validation_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga tb $name]
    }
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_late_validation_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_late_validation_r1_bad_${mutant}.sv]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_late_validation_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_late_validation_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_LATE_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project symbolic_late_validation_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {}} {
        if {$caught || [string first "LATE_VALIDATION_PASS bank=$mode rows=$rows samples=[expr {$rows+4}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Late validation failed: $mode $message"
        }
    } elseif {[string first "LATE_VALIDATION_PASS" $text]>=0 || ![regexp {Fatal: LATE_FAULT_MISMATCH row=} $text]} {
        error "Late validation mutant not rejected: $mode"
    }
    close_project
    puts "NTRPU_LATE_CASE_REVIEWED $mode"
    return $text
}
proc late_validation {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set control_banks [dict create]
    foreach item {
        {main pcie_request_inventory_r7.txt 22339}
        {ownership pcie_rc_ownership_integration.txt 46356}
        {rq pcie_rq_admission_integration.txt 8850}
        {prefix pcie_rc_prefix_integration.txt 544}
        {lanes pcie_rc_lane_integration_rtl.txt 984}
        {context pcie_symbolic_value_context.txt 1047}
        {drained_main pcie_symbolic_numeric_ab_drained_main_r1.txt 22607}
        {drained_ownership pcie_symbolic_numeric_ab_drained_ownership_r1.txt 61840}
        {drained_rq pcie_symbolic_numeric_ab_drained_rq_r1.txt 9806}
        {drained_prefix pcie_symbolic_numeric_ab_drained_prefix_r1.txt 780}
        {drained_lanes pcie_symbolic_numeric_ab_drained_lanes_r1.txt 1348}
        {drained_context pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {directed pcie_symbolic_ced_directed_r1.txt 217}
        {controls pcie_symbolic_numeric_ab_controls_r1.txt 352}
    } {
        lassign $item bank vector rows
        set text [late_case $output $bank $vector $rows {}]
        foreach bit {4 7 8 9 10} {
            if {![regexp [format {(?m)^LATE_FAULT_COVERAGE bit=%d hits=([0-9]+)$} $bit] $text all hits]} {error "Missing fault coverage: $bank $bit"}
            if {$hits>0 && ![dict exists $control_banks $bit]} {dict set control_banks $bit $item}
        }
    }
    foreach bit {4 7 8 9 10} {
        if {![dict exists $control_banks $bit]} {error "No positive late-fault coverage for bit $bit"}
        lassign [dict get $control_banks $bit] bank vector rows
        puts "NTRPU_LATE_CONTROL_BANK mode=bad_fault$bit bank=$bank rows=$rows"
        late_case $output bad_fault$bit $vector $rows fault$bit
    }
    puts NTRPU_SYMBOLIC_LATE_VALIDATION_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {late_validation [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
