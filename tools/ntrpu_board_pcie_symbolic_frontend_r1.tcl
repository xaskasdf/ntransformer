set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_frontend_r1 {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    set banks {
        {main pcie_request_inventory_r7.txt 22339}
        {ownership pcie_rc_ownership_integration.txt 46356}
        {rq pcie_rq_admission_integration.txt 8850}
        {prefix pcie_rc_prefix_integration.txt 544}
        {lanes pcie_rc_lane_integration_rtl.txt 984}
        {context pcie_symbolic_value_context.txt 1047}
    }
    set cases $banks
    foreach mode {omit_eop reload sequence cold} {
        lappend cases [list $mode pcie_symbolic_value_context.txt 1047]
    }
    foreach item $cases {
        lassign $item mode vector rows
        set folder [file join $output $mode]
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector]]
        create_project symbolic_s1 [file join $folder project] -part xcvu5p-flva2104-2-i
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv} {
            add_files [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        set positive [expr {[lsearch -exact {omit_eop reload sequence cold} $mode]<0}]
        if {$positive} {
            add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_frontend_r1.sv]
        } else {
            add_files [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_frontend_r1_bad_${mode}.sv]
        }
        foreach name {ntrpu_pcie_inventory_symbolic_model_r1.sv ntrpu_pcie_request_inventory_symbolic_probe.sv} {
            add_files [file join $ntrpu_script_dir .. fpga tb $name]
        }
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_frontend_r1.sv]
        set_property top tb_ntrpu_pcie_symbolic_frontend_r1 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows" [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_S1_CASE_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project symbolic_s1.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$positive} {
            if {$caught || [string first "S1_FRONTEND_PASS bank=$mode rows=$rows " $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
                error "S1 boundary failed: $mode $message"
            }
        } else {
            set expected [dict get {omit_eop S1_EOP_ELIGIBILITY_MISMATCH reload S1_EOP_LANES_MISMATCH sequence S1_RECORD_MISMATCH cold S1_COLD_CAPTURE_MISMATCH} $mode]
            if {[string first "S1_FRONTEND_PASS" $text]>=0 || [string first "Fatal: $expected row=" $text]<0} {
                error "S1 actual mutant not rejected: $mode"
            }
        }
        close_project
        puts "NTRPU_S1_CASE_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_FRONTEND_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_frontend_r1 [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
