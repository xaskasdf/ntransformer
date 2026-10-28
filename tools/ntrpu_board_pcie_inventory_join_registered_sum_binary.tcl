set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_join_binary {mode input output} {
    global ntrpu_script_dir
    if {$mode ni {rtl netlist} || [file exists $output]} {error "Invalid mode/output"}
    file mkdir $output
    set_param general.maxThreads 2
    if {$mode eq "netlist"} {
        set seed [file join $input pcie_inventory_join_funcsim.v]
        if {![file isfile $seed]} {error "Missing reviewed functional netlist"}
        file copy $seed [file join $output input_funcsim.v]
    }
    create_project inventory_join_binary [file join $output project] -part xcvu5p-flva2104-2-i
    if {$mode eq "rtl"} {
        foreach name {ntrpu_pcie_inventory_join.sv ntrpu_pcie_inventory_scheduler_r14.sv ntrpu_pcie_inventory_sum_accumulator.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_inventory_validation.sv} {
            add_files [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        set_property verilog_define JOIN_RTL_INTERNAL_CHECKS [get_filesets sim_1]
    } else {
        add_files [file join $output input_funcsim.v]
    }
    set_property top ntrpu_pcie_inventory_join [get_filesets sources_1]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_request_inventory_r7.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_join_binary.sv]
    set_property top tb_ntrpu_pcie_inventory_join [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_binary_r5.txt]]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    if {[lsearch -nocase -exact [list_property [get_filesets sim_1]] xsim.simulate.custom_tcl]<0} {error "Missing custom_tcl property"}
    set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
    if {[get_property xsim.simulate.custom_tcl [get_filesets sim_1]] ne $startup} {error "Startup property mismatch"}
    puts "NTRPU_XSIM_CUSTOM_STARTUP $startup"
    update_compile_order -fileset sim_1
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_prefix_integration_binary.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_rq_integration_binary.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_ownership_integration_binary.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_lane_integration_binary.sv]
    set lane_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_lane_integration_binary.txt]]
    set_property top tb_ntrpu_pcie_inventory_lane_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$lane_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output project inventory_join_binary.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output lane_integration]
    file copy $logfile [file join $output lane_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=632 scenarios=60 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RC lane integration failed: $message"}
    puts NTRPU_INVENTORY_RC_LANE_INTEGRATION_REVIEWED
    set ownership_integration_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_ownership_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_ownership_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$ownership_integration_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output project inventory_join_binary.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output ownership_integration]
    file copy $logfile [file join $output ownership_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=46356 scenarios=3872 mode=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RC ownership integration failed: $message"}
    puts "NTRPU_INVENTORY_OWNERSHIP_INTEGRATION_REVIEWED mode=$mode"
    set rq_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rq_admission_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_rq_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$rq_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output project inventory_join_binary.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output rq_integration]
    file copy $logfile [file join $output rq_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=8850 scenarios=240 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RQ raw integration failed: $message"}
    puts "NTRPU_INVENTORY_RQ_INTEGRATION_REVIEWED mode=$mode"
    set prefix_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_prefix_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_prefix_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$prefix_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output project inventory_join_binary.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output prefix_integration]
    file copy $logfile [file join $output prefix_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=544 scenarios=60 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Prefix binary integration failed: $message"}
    puts "NTRPU_INVENTORY_PREFIX_INTEGRATION_REVIEWED mode=$mode"
    set_property top tb_ntrpu_pcie_inventory_join [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output project inventory_join_binary.sim sim_1 behav xsim simulate.log]
    file mkdir [file join $output normal]
    file copy $logfile [file join $output normal simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    if {$caught || [string first "JOIN_SUITE_PASS rows=22330 scenarios=67 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Binary scheduler failed: $mode $message"}
    close_project
    puts "NTRPU_PCIE_INVENTORY_JOIN_BINARY_EXPORTED mode=$mode"
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_join_binary [lindex $argv 0] [file normalize [lindex $argv 1]] [file normalize [lindex $argv 2]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
