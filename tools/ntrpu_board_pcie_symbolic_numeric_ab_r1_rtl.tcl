set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_numeric_ab_r1_rtl {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
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
        {bad_swap_fields pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_drop_unowned_context pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_omit_origin pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_wrong_fault pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_ignore_enable pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_omit_terminal pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
        {bad_omit_cold pcie_symbolic_numeric_ab_drained_context_r1.txt 1115}
    } {
        lassign $item mode vector rows
        set folder [file join $output $mode]
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        create_project symbolic_numeric_ab_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
        set sources {}
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv} {
            lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
            lappend sources [file join $ntrpu_script_dir .. fpga tb $name]
        }
        foreach unit {apply delivery} {
            set candidate [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_${unit}_r1.sv]
            set mutant [string range $mode 4 end]
            if {([string match "bad_*" $mode]) && (($unit eq "apply" && $mutant in {swap_fields drop_unowned_context omit_origin wrong_fault}) ||
                ($unit eq "delivery" && $mutant in {ignore_enable omit_terminal omit_cold}))} {
                set candidate [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_numeric_ab_r1_bad_${mutant}.sv]
            }
            lappend sources $candidate
        }
        add_files $sources
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_numeric_ab_r1.sv]
        set_property top tb_ntrpu_pcie_symbolic_numeric_ab_r1 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_NUMERIC_AB_CASE_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project symbolic_numeric_ab_r1.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {[string match "bad_*" $mode]} {
            if {[string first "NUMERIC_AB_PASS" $text]>=0 || ![regexp {Fatal: NUMERIC_[A-Z_]+MISMATCH row=} $text]} {
                error "Numeric mutant not rejected: $mode"
            }
        } elseif {$caught || [string first "NUMERIC_AB_PASS bank=$mode rows=$rows " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Numeric A/B boundary failed: $mode $message"
        }
        close_project
        puts "NTRPU_NUMERIC_AB_CASE_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_NUMERIC_AB_R1_RTL_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_numeric_ab_r1_rtl [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
