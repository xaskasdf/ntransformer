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
        foreach name {ntrpu_pcie_inventory_join.sv ntrpu_pcie_inventory_scheduler_r4.sv ntrpu_pcie_rc_update_select.sv ntrpu_pcie_inventory_validation.sv} {
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
