set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc monitor_case {output mode vector rows mutant directed} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
    create_project symbolic_monitor_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv ntrpu_pcie_symbolic_late_validation_r1.sv ntrpu_pcie_symbolic_retirement_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv} {
        lappend sources [file join $ntrpu_script_dir .. fpga tb $name]
    }
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_monitor_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_monitor_r1_bad_${mutant}.sv]
    }
    add_files $sources
    set tb tb_ntrpu_pcie_symbolic_monitor_r2
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${tb}.sv]
    set_property top $tb [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"
    if {$directed} {append options " -testplusarg QUIET_DIRECTED"}
    set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_MONITOR_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project symbolic_monitor_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {}} {
        if {$caught || [string first "MONITOR_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "RETIREMENT_PASS bank=$mode rows=$rows samples=[expr {$rows+5}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Monitor failed: $mode $message"
        }
    } elseif {[string first "MONITOR_PASS" $text]>=0 || ![regexp {Fatal: MONITOR_(PENDING|EMPTY|PROBE)_MISMATCH row=} $text]} {
        error "Monitor mutant not rejected at intended comparator: $mode"
    }
    close_project
    puts "NTRPU_MONITOR_CASE_REVIEWED $mode"
    return $text
}
proc monitor_directed {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    monitor_case $output monitor pcie_symbolic_monitor_directed_r2.txt 252 {} 1
    foreach mutant {omit_c omit_e omit_d omit_a omit_b extra_retired omit_early_fault omit_late_fault omit_cold omit_observing omit_requester omit_attributes omit_rq_ready omit_rc_ready omit_raw_rq omit_raw_rc} {
        monitor_case $output bad_$mutant pcie_symbolic_monitor_directed_r2.txt 252 $mutant 1
    }
    foreach item {
        {directed pcie_symbolic_ced_directed_r1.txt 217}
        {controls pcie_symbolic_numeric_ab_controls_r1.txt 352}
        {range_return pcie_symbolic_retirement_range_r1.txt 277}
    } {
        lassign $item bank vector rows
        monitor_case $output $bank $vector $rows {} 0
    }
    puts NTRPU_SYMBOLIC_MONITOR_DIRECTED_R4_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {monitor_directed [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
