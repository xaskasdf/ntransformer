# Identical binary shadow/boundary benches for RTL and mapped functional replay.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc rc_binary_suite {mode input output} {
    global ntrpu_script_dir
    if {$mode ni {rtl netlist} || [file exists $output]} {error "Invalid mode/output"}
    file mkdir $output
    set_param general.maxThreads 2
    if {$mode eq "netlist"} {
        set seed [file join $input pcie_rc_event_stage_funcsim.v]
        if {![file isfile $seed]} {error "Missing reviewed functional netlist"}
        file copy $seed [file join $output input_funcsim.v]
    }
    foreach bank {shadow boundaries fragments} {
        set folder [file join $output $bank]
        create_project rc_event_binary $folder -part xcvu5p-flva2104-2-i
        if {$mode eq "rtl"} {
            foreach name {ntrpu_pcie_rc_proposal.sv ntrpu_pcie_rc_event_stage_r6.sv} {
                add_files [file join $ntrpu_script_dir .. fpga rtl $name]
            }
        } else {
            add_files [file join $output input_funcsim.v]
        }
        set_property top ntrpu_pcie_rc_event_stage [get_filesets sources_1]
        if {$bank eq "shadow"} {
            foreach name {ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_rc_proposal_reference.sv} {
                add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl $name]
            }
            set top tb_ntrpu_pcie_rc_event_stage_binary
            set vector_name pcie_request_inventory_binary_r5.txt
            set marker "RC_STAGE_SUITE_PASS rows=22330 scenarios=67 mode=0 latency=2"
        } elseif {$bank eq "boundaries"} {
            set top tb_ntrpu_pcie_rc_event_boundaries
            set vector_name pcie_rc_event_boundaries.txt
            set marker "RC_BOUNDARY_SUITE_COMPLETE rows=370 phases=15 latency=2 native=0"
        } else {
            foreach name {ntrpu_pcie_rc_event_stage_r1_reference.sv ntrpu_pcie_rc_proposal_reference.sv} {
                add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl $name]
            }
            set top tb_ntrpu_pcie_rc_fragment_matrix
            set vector_name pcie_rc_fragment_matrix.txt
            set marker "RC_FRAGMENT_MATRIX_REFERENCE_PASS rows=281904 native=0"
        }
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data $vector_name]]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set f [open [file join $folder rc_event_binary.sim sim_1 behav xsim simulate.log] r]
        set text [read $f];close $f
        if {$caught || [string first $marker $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Binary suite failed: $bank $mode $message"
        }
        close_project
        puts "NTRPU_RC_EVENT_BINARY_BANK_COMPLETE mode=$mode bank=$bank"
    }
    puts "NTRPU_PCIE_RC_EVENT_BINARY_EXPORTED mode=$mode"
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {rc_binary_suite [lindex $argv 0] [file normalize [lindex $argv 1]] [file normalize [lindex $argv 2]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
