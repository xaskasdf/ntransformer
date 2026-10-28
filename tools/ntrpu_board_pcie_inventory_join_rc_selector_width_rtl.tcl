set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_join_rtl {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project inventory_join_rtl $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_inventory_scheduler_r9.sv ntrpu_pcie_rc_update_select_r4.sv ntrpu_pcie_inventory_validation.sv ntrpu_pcie_inventory_join.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_join.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_update_select_r1.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_prefix_math.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_prefix_integration.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rq_admission.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_rq_integration.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_event_ancestry.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_rc_update_select_reference_r1.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_inventory_ownership_integration.sv]
    set_property top tb_ntrpu_pcie_inventory_join [get_filesets sim_1]
    set_property verilog_define JOIN_RTL_INTERNAL_CHECKS [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_r7.txt]]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    if {[lsearch -nocase -exact [list_property [get_filesets sim_1]] xsim.simulate.custom_tcl]<0} {error "Missing custom_tcl property"}
    set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
    if {[get_property xsim.simulate.custom_tcl [get_filesets sim_1]] ne $startup} {error "Startup property mismatch"}
    puts "NTRPU_XSIM_CUSTOM_STARTUP $startup"
    update_compile_order -fileset sim_1
    set ownership_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_ownership.txt]]
    set_property top tb_ntrpu_pcie_rc_event_ancestry [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg OWNERSHIP_VECTORS=$ownership_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output ownership]
    file copy $logfile [file join $output ownership simulate.log]
    if {$caught || [string first "RC_OWNERSHIP_PASS rows=155168 checks=1153296 unknown_checks=994288" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RC ownership failed: $message"}
    if {[string first "RC_EVENT_ANCESTRY_PASS checks=32768 unknown_checks=32256" $text]<0} {error "Missing event ancestry controls"}
    puts NTRPU_INVENTORY_RC_OWNERSHIP_REVIEWED
    set ownership_integration_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_ownership_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_ownership_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$ownership_integration_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output ownership_integration]
    file copy $logfile [file join $output ownership_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=46356 scenarios=3872 mode=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RC ownership integration failed: $message"}
    puts "NTRPU_INVENTORY_OWNERSHIP_INTEGRATION_REVIEWED mode=full_rtl"
    set_property top tb_ntrpu_pcie_rq_admission [get_filesets sim_1]
    set_property xsim.simulate.xsim.more_options {} [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output rq_admission]
    file copy $logfile [file join $output rq_admission simulate.log]
    if {$caught || [string first "RQ_ADMISSION_PASS checks=615536 unknown_checks=5712" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RQ admission failed: $message"}
    puts NTRPU_INVENTORY_RQ_ADMISSION_REVIEWED
    set rq_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rq_admission_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_rq_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$rq_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output rq_integration]
    file copy $logfile [file join $output rq_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=8850 scenarios=240 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RQ raw integration failed: $message"}
    puts "NTRPU_INVENTORY_RQ_INTEGRATION_REVIEWED mode=full_rtl"
    set_property top tb_ntrpu_pcie_rc_update_select_r1 [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set select_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_update_ancestry.txt]]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SELECT_VECTORS=$select_vectors"] [get_filesets sim_1]
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output selection]
    file copy $logfile [file join $output selection simulate.log]
    if {$caught || [string first "RC_UPDATE_ANCESTRY_PASS rows=89632" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "RC selection failed: $message"}
    puts NTRPU_INVENTORY_JOIN_RC_ANCESTRY_REVIEWED

    set_property top tb_ntrpu_pcie_rc_prefix_math [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output arithmetic]
    file copy $logfile [file join $output arithmetic simulate.log]
    if {$caught || [string first "RC_PREFIX_MATH_PASS checks=45056 unknown_checks=6624" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Prefix arithmetic failed: $message"}
    puts NTRPU_INVENTORY_JOIN_RC_PREFIX_MATH_REVIEWED
    set prefix_vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_prefix_integration.txt]]
    set_property top tb_ntrpu_pcie_inventory_prefix_integration [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$prefix_vectors"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $output inventory_join_rtl.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file mkdir [file join $output prefix_integration]
    file copy $logfile [file join $output prefix_integration simulate.log]
    if {$caught || [string first "JOIN_SUITE_PASS rows=544 scenarios=60 mode=0" $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Prefix raw integration failed: $message"}
    puts "NTRPU_INVENTORY_PREFIX_INTEGRATION_REVIEWED mode=full_rtl"
    set_property top tb_ntrpu_pcie_inventory_join [get_filesets sim_1]
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
