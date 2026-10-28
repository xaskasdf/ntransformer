set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_ced_r1 {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    foreach item {
        {context pcie_symbolic_value_context.txt 1047}
        {lanes pcie_rc_lane_integration_rtl.txt 984}
        {prefix pcie_rc_prefix_integration.txt 544}
        {main pcie_request_inventory_r7.txt 22339}
        {ownership pcie_rc_ownership_integration.txt 46356}
        {rq pcie_rq_admission_integration.txt 8850}
    } {
        lassign $item mode vector rows
        set folder [file join $output $mode]
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        create_project symbolic_ced [file join $folder project] -part xcvu5p-flva2104-2-i
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r2.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r2.sv} {
            add_files [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
            add_files [file join $ntrpu_script_dir .. fpga tb $name]
        }
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_ced_r1.sv]
        set_property top tb_ntrpu_pcie_symbolic_ced_r1 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_CED_CASE_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project symbolic_ced.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$caught || [string first "CED_FRONTEND_PASS bank=$mode rows=$rows " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "CED boundary failed: $mode $message"
        }
        close_project
        puts "NTRPU_CED_CASE_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_CED_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_ced_r1 [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
