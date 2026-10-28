set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc rc_event_stage {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project rc_event_stage $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_rc_proposal.sv ntrpu_pcie_rc_event_stage.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_event_stage.sv]
    set_property top tb_ntrpu_pcie_rc_event_stage [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_r7.txt]]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach mode {normal omit_rc missing_metadata} {
        set args "-testplusarg VECTORS=$vectors"
        if {$mode eq "omit_rc"} {append args " -testplusarg OMIT_RC"}
        if {$mode eq "missing_metadata"} {append args " -testplusarg MISSING_METADATA"}
        set_property -dict [list xsim.simulate.xsim.more_options $args] [get_filesets sim_1]
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set target [file join $output $mode]
        file mkdir $target
        set logfile [file join $output rc_event_stage.sim sim_1 behav xsim simulate.log]
        file copy $logfile [file join $target simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        if {$mode eq "normal"} {
            if {$caught || [string first "RC_STAGE_SUITE_PASS rows=22339 scenarios=68 mode=0 latency=2" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Normal metadata shadow failed: $message"}
        } else {
            if {[string first "RC_STAGE_SUITE_PASS" $text]>=0 || ![regexp {Fatal:.*RC_STAGE_(EVENT|CONTEXT|VALIDATION)_MISMATCH} $text]} {error "Expected shadow mutation rejection missing: $mode"}
        }
        puts "NTRPU_RC_STAGE_CASE_REVIEWED $mode"
    }
    close_project
    puts NTRPU_PCIE_RC_EVENT_STAGE_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {rc_event_stage [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
