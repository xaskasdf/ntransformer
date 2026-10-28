set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_ced_control_r3_rtl {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    foreach mode {positive early_limit omit_second unknown_low} {
        set folder [file join $output capacity_$mode]
        create_project capacity_r3 [file join $folder project] -part xcvu5p-flva2104-2-i
        set top tb_ntrpu_pcie_symbolic_capacity_r3_$mode
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_CED_R3_CAPACITY_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project capacity_r3.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$mode eq "positive"} {
            if {$caught || [string first "CAPACITY_COMPLETE_PASS predicates=16777216 admissions=2098124 supplementary=972" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
                error "Capacity proof failed: $message"
            }
        } else {
            if {[string first "CAPACITY_COMPLETE_PASS" $text]>=0 || ![regexp {Fatal: CAPACITY_PREDICATE_MISMATCH case=} $text]} {
                error "Capacity mutant not rejected: $mode"
            }
        }
        close_project
        puts "NTRPU_CED_R3_CAPACITY_REVIEWED $mode"
    }
    foreach item {
        {main pcie_request_inventory_r7.txt 22339}
        {ownership pcie_rc_ownership_integration.txt 46356}
        {rq pcie_rq_admission_integration.txt 8850}
        {prefix pcie_rc_prefix_integration.txt 544}
        {lanes pcie_rc_lane_integration_rtl.txt 984}
        {context pcie_symbolic_value_context.txt 1047}
        {drained_main pcie_symbolic_ced_drained_main_r1.txt 22473}
        {drained_ownership pcie_symbolic_ced_drained_ownership_r1.txt 54098}
        {drained_rq pcie_symbolic_ced_drained_rq_r1.txt 9328}
        {drained_prefix pcie_symbolic_ced_drained_prefix_r1.txt 662}
        {drained_lanes pcie_symbolic_ced_drained_lanes_r1.txt 1166}
        {drained_context pcie_symbolic_ced_drained_context_r1.txt 1081}
        {directed pcie_symbolic_ced_directed_r1.txt 217}
    } {
        lassign $item mode vector rows
        set folder [file join $output $mode]
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        create_project symbolic_ced_control_r3 [file join $folder project] -part xcvu5p-flva2104-2-i
        set sources {}
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv} {
            lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
            lappend sources [file join $ntrpu_script_dir .. fpga tb $name]
        }
        add_files $sources
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_ced_control_r3.sv]
        set_property top tb_ntrpu_pcie_symbolic_ced_control_r3 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_CED_R3_CASE_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project symbolic_ced_control_r3.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$caught || [string first "CED_FRONTEND_PASS bank=$mode rows=$rows " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "R3 full boundary failed: $mode $message"
        }
        close_project
        puts "NTRPU_CED_R3_CASE_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_CED_CONTROL_R3_RTL_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_ced_control_r3_rtl [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
