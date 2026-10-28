set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc symbolic_value_context {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_symbolic_value_context.txt]]
    foreach mode {normal reload delivery attributes} {
        set folder [file join $output $mode]
        create_project symbolic_value [file join $folder project] -part xcvu5p-flva2104-2-i
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_inventory_validation.sv ntrpu_pcie_inventory_join.sv} {
            add_files [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        set model ntrpu_pcie_inventory_symbolic_model_r1.sv
        if {$mode ne "normal"} {set model ntrpu_pcie_inventory_symbolic_model_r1_bad_${mode}.sv}
        add_files [file join $ntrpu_script_dir .. fpga tb $model]
        add_files [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_request_inventory_symbolic_probe.sv]
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_value_context.sv]
        set_property top tb_ntrpu_pcie_symbolic_value_context [get_filesets sim_1]
        set_property verilog_define JOIN_RTL_INTERNAL_CHECKS [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_SYMBOLIC_VALUE_CASE_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project symbolic_value.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$mode eq "normal"} {
            if {$caught || [string first "SYMBOLIC_VALUE_CONTEXT_PASS rows=1047 observations=1048 scenarios=18 query_bits=180 delta_bits=141 rc_bits=71 rq_bits=43" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Symbolic value context failed: $message"}
        } else {
            if {[string first "SYMBOLIC_VALUE_CONTEXT_PASS" $text]>=0 ||
                ![regexp {Fatal: (SYMBOLIC_(QUERY|DELTA|RETAINED)_MISMATCH|JOIN_TABLE_MISMATCH) row=[0-9]+} $text]} {
                error "Actual symbolic value mutant not rejected: $mode"
            }
        }
        close_project
        puts "NTRPU_SYMBOLIC_VALUE_CASE_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_VALUE_CONTEXT_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {symbolic_value_context [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
