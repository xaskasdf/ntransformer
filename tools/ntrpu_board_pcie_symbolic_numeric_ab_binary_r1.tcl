set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_numeric_binary {seed output} {
    global ntrpu_script_dir
    if {[file exists $output] || ![file isfile $seed]} {error "Invalid seed/output"}
    file mkdir $output
    file copy $seed [file join $output input_funcsim.v]
    set_param general.maxThreads 2
    create_project symbolic_numeric_binary [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files [file join $output input_funcsim.v]
    foreach name {ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_symbolic_frontend_r1.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
        add_files [file join $ntrpu_script_dir .. fpga tb numeric_ab_mapped_refs_r1 $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_numeric_ab_binary_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_numeric_ab_binary_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set cases {
        {controls pcie_symbolic_numeric_ab_controls_r1.txt 352 {}}
        {omit_control pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_CONTROL}
        {omit_record pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_RECORD}
        {omit_cold pcie_symbolic_numeric_ab_controls_r1.txt 352 OMIT_COLD}
        {wrong_origin pcie_symbolic_numeric_ab_controls_r1.txt 352 WRONG_ORIGIN}
        {ignore_enable pcie_symbolic_numeric_ab_controls_r1.txt 352 IGNORE_ENABLE}
        {raw_main pcie_symbolic_ced_binary_raw_main_r1.txt 18538 {}}
        {raw_ownership pcie_rc_ownership_integration.txt 46356 {}}
        {raw_rq pcie_rq_admission_integration.txt 8850 {}}
        {raw_prefix pcie_rc_prefix_integration.txt 544 {}}
        {raw_lanes pcie_rc_lane_integration_binary.txt 632 {}}
        {raw_context pcie_symbolic_value_context_binary.txt 997 {}}
        {drained_main pcie_symbolic_numeric_ab_binary_drained_main_r1.txt 18794 {}}
        {drained_ownership pcie_symbolic_numeric_ab_drained_ownership_r1.txt 61840 {}}
        {drained_rq pcie_symbolic_numeric_ab_drained_rq_r1.txt 9806 {}}
        {drained_prefix pcie_symbolic_numeric_ab_drained_prefix_r1.txt 780 {}}
        {drained_lanes pcie_symbolic_numeric_ab_binary_drained_lanes_r1.txt 872 {}}
        {drained_context pcie_symbolic_numeric_ab_binary_drained_context_r1.txt 1049 {}}
        {directed pcie_symbolic_ced_binary_directed_r1.txt 191 {}}
    }
    foreach item $cases {
        lassign $item bank vector rows mutation
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$bank -testplusarg ROWS=$rows"
        if {$mutation ne {}} {append options " -testplusarg $mutation"}
        set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
        puts "NTRPU_NUMERIC_MAPPED_CASE_START $bank"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set sim [file join $output project symbolic_numeric_binary.sim sim_1 behav xsim]
        set folder [file join $output $bank];file mkdir $folder
        foreach file [glob -nocomplain -directory $sim *.log *.prj *.sh *.tcl] {file copy $file $folder}
        set f [open [file join $folder simulate.log] r];set text [read $f];close $f
        if {$mutation eq {}} {
            if {$caught || [string first "NUMERIC_MAPPED_PASS bank=$bank rows=$rows samples=[expr {$rows+4}] mode=0 ports=2892" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Mapped numeric mismatch: $bank $message"}
        } else {
            if {[string first "NUMERIC_MAPPED_PASS" $text]>=0 || [string first "Fatal: NUMERIC_MAPPED_PORT_MISMATCH row=" $text]<0} {
                error "Mapped boundary control not rejected: $bank"
            }
        }
        puts "NTRPU_NUMERIC_MAPPED_CASE_REVIEWED $bank"
    }
    close_project
    puts NTRPU_SYMBOLIC_NUMERIC_MAPPED_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_numeric_binary [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
