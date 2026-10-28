set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc retirement_case {output mode vector rows mutant targeted} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
    create_project symbolic_retirement_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv ntrpu_pcie_symbolic_late_validation_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga tb $name]
    }
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_retirement_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_retirement_r1_bad_${mutant}.sv]
    }
    add_files $sources
    set tb tb_ntrpu_pcie_symbolic_retirement_r1
    if {$targeted} {set tb tb_ntrpu_pcie_symbolic_retirement_range_r1}
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${tb}.sv]
    set_property top $tb [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_RETIRE_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project symbolic_retirement_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {}} {
        if {$caught || [string first "RETIREMENT_PASS bank=$mode rows=$rows samples=[expr {$rows+5}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Retirement failed: $mode $message"
        }
    } elseif {[string first "RETIREMENT_PASS" $text]>=0 || ![regexp {Fatal: RETIRE_[A-Z_]+MISMATCH row=} $text]} {
        error "Retirement mutant not rejected: $mode"
    }
    close_project
    puts "NTRPU_RETIRE_CASE_REVIEWED $mode"
    return $text
}
proc retirement_full {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set text [retirement_case $output range_return pcie_symbolic_retirement_range_r1.txt 277 {} 1]
    if {[string first "RANGE_RETURN_PASS reference_checks=4 retirement_checks=4 cold_checks=1" $text]<0} {error "Range-return witnesses missing"}
    set text [retirement_case $output bad_drop_byte_sticky pcie_symbolic_retirement_range_r1.txt 277 drop_byte_sticky 1]
    if {![regexp {Fatal: RETIRE_STATE_MISMATCH row=265 origin=260} $text]} {error "Wrong byte-sticky mutant origin"}
    if {![regexp {RETIRE_EXPECTED ([0-9a-fA-F]+)} $text all wanted] || ![regexp {RETIRE_ACTUAL ([0-9a-fA-F]+)} $text all actual]} {error "Missing byte-sticky states"}
    set wanted_number [expr 0x$wanted];set actual_number [expr 0x$actual]
    if {($wanted_number ^ $actual_number)!=8192} {error "Byte-sticky mutant did not differ exclusively in bit13"}
    set text [retirement_case $output bad_keep_fault_cold pcie_symbolic_retirement_range_r1.txt 277 keep_fault_cold 1]
    if {![regexp {Fatal: RETIRE_COLD_MISMATCH row=272} $text]} {error "Cold-sticky mutant did not fail at busy cold boundary"}
    puts NTRPU_RETIRE_RANGE_RETURN_GATE_PASSED
    foreach item {
        {main pcie_request_inventory_r7.txt 22339}
        {ownership pcie_rc_ownership_integration.txt 46356}
        {rq pcie_rq_admission_integration.txt 8850}
        {prefix pcie_rc_prefix_integration.txt 544}
        {lanes pcie_rc_lane_integration_rtl.txt 984}
        {context pcie_symbolic_value_context.txt 1047}
        {drained_ownership pcie_symbolic_retirement_drained_ownership_r1.txt 65711}
        {drained_rq pcie_symbolic_retirement_drained_rq_r1.txt 10045}
        {drained_prefix pcie_symbolic_retirement_drained_prefix_r1.txt 839}
        {drained_lanes pcie_symbolic_retirement_drained_lanes_r1.txt 1439}
        {drained_context pcie_symbolic_retirement_drained_context_r1.txt 1132}
    } {
        lassign $item bank vector rows
        retirement_case $output $bank $vector $rows {} 0
    }
    foreach item {
        {wrong_origin pcie_symbolic_numeric_ab_controls_r1.txt 352 0}
        {ignore_enable pcie_symbolic_numeric_ab_controls_r1.txt 352 0}
        {double_add pcie_symbolic_numeric_ab_controls_r1.txt 352 0}
        {omit_cold pcie_symbolic_numeric_ab_controls_r1.txt 352 0}
        {omit_sticky pcie_symbolic_retirement_range_r1.txt 277 1}
        {omit_high pcie_symbolic_retirement_drained_main_r1.txt 22674 0}
        {wide_feedback pcie_symbolic_retirement_drained_main_r1.txt 22674 0}
    } {
        lassign $item mutant vector rows targeted
        retirement_case $output bad_$mutant $vector $rows $mutant $targeted
    }
    puts NTRPU_SYMBOLIC_RETIREMENT_FULL_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retirement_full [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
