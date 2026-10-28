set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_s1_binary {seed output} {
    global ntrpu_script_dir
    if {[file exists $output] || ![file isfile $seed]} {error "Invalid seed/output"}
    file mkdir $output
    file copy $seed [file join $output input_funcsim.v]
    set_param general.maxThreads 2
    create_project symbolic_s1_binary [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files [file join $output input_funcsim.v]
    foreach name {ntrpu_pcie_symbolic_frontend_r1_reference.sv ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
        add_files [file join $ntrpu_script_dir .. fpga tb $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_frontend_r1_binary.sv]
    set_property top tb_ntrpu_pcie_symbolic_frontend_r1_binary [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set cases {
        {main pcie_request_inventory_binary_r5.txt 22330 {}}
        {ownership pcie_rc_ownership_integration.txt 46356 {}}
        {rq pcie_rq_admission_integration.txt 8850 {}}
        {prefix pcie_rc_prefix_integration.txt 544 {}}
        {lanes pcie_rc_lane_integration_binary.txt 632 {}}
        {context pcie_symbolic_value_context_binary.txt 997 {}}
        {omit_rq pcie_symbolic_value_context_binary.txt 997 OMIT_RQ}
        {omit_rc pcie_symbolic_value_context_binary.txt 997 OMIT_RC}
        {omit_cold pcie_symbolic_value_context_binary.txt 997 OMIT_COLD}
        {wrong_sequence pcie_symbolic_value_context_binary.txt 997 WRONG_SEQUENCE}
    }
    foreach item $cases {
        lassign $item bank vector rows mutation
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$bank -testplusarg ROWS=$rows"
        if {$mutation ne {}} {append options " -testplusarg $mutation"}
        set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
        puts "NTRPU_S1_MAPPED_CASE_START $bank"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set sim [file join $output project symbolic_s1_binary.sim sim_1 behav xsim]
        set folder [file join $output $bank];file mkdir $folder
        foreach file [glob -nocomplain -directory $sim *.log *.prj *.sh *.tcl] {file copy $file $folder}
        set f [open [file join $folder simulate.log] r];set text [read $f];close $f
        if {$mutation eq {}} {
            if {$caught || [string first "S1_MAPPED_PASS bank=$bank rows=$rows mode=0 ports=2498" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Mapped S1 mismatch: $bank $message"}
        } else {
            if {[string first "S1_MAPPED_PASS" $text]>=0 || [string first "Fatal: S1_MAPPED_PORT_MISMATCH row=" $text]<0} {
                error "Mapped boundary control not rejected: $bank"
            }
        }
        puts "NTRPU_S1_MAPPED_CASE_REVIEWED $bank"
    }
    close_project
    puts NTRPU_SYMBOLIC_FRONTEND_MAPPED_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_s1_binary [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
