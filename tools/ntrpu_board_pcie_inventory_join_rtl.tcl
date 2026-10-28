set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_join_rtl {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project inventory_join_rtl $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_inventory_scheduler.sv ntrpu_pcie_inventory_validation.sv ntrpu_pcie_inventory_join.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_join.sv]
    set_property top tb_ntrpu_pcie_inventory_join [get_filesets sim_1]
    set_property verilog_define JOIN_RTL_INTERNAL_CHECKS [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_r7.txt]]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach mode {normal omit_rq omit_rc omit_validation false_empty wrong_sequence omit_cold_reset corrupt_metadata} {
        set args "-testplusarg VECTORS=$vectors"
        if {$mode ne "normal"} {append args " -testplusarg [string toupper $mode]"}
        set_property -dict [list xsim.simulate.xsim.more_options $args] [get_filesets sim_1]
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set target [file join $output $mode]
        file mkdir $target
        set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
        file copy $logfile [file join $target simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        if {$mode eq "normal"} {
            if {$caught || [string first "JOIN_SUITE_PASS rows=22339 scenarios=68 mode=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Integrated inventory failed: $message"}
        } else {
            if {[string first "JOIN_SUITE_PASS" $text]>=0 || ![regexp {Fatal: JOIN_(STATE|SEQUENCE|PENDING|EMPTY|REVOKE|FAULT)_MISMATCH} $text]} {error "Actual join mutation was not rejected: $mode"}
        }
        puts "NTRPU_INVENTORY_JOIN_CASE_REVIEWED $mode"
    }
    close_project
    puts NTRPU_PCIE_INVENTORY_JOIN_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_join_rtl [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
