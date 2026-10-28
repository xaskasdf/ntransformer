set root [file normalize [file join [file dirname [info script]] ..]]
proc characterize_join {root output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set source [file join $root fpga rtl ntrpu_drain_certificate_join.sv]
    set f [open $source r];set original [read $f];close $f
    set mutations [list \
        [list omit_local {if(current && (&good))} {if(current && (&good[4:0]))} 1 {Fatal: JOIN_CHECK_FAILED case=10 label=1}] \
        [list stale_high {&& armed[i] && ack_valid[i]} {&& ack_valid[i]} 2 {Fatal: JOIN_CHECK_FAILED case=10 label=11}] \
        [list forget_fault {faults<=faults | proof_fault;} {faults<=0;} 1 {Fatal: JOIN_FAULT_NOT_RETAINED}] \
        [list ignore_tag {ack_tag[i*64+:64]==saved_tag} {1'b1} 2 {Fatal: JOIN_CHECK_FAILED case=30 label=30}]]
    foreach mutation $mutations {
        lassign $mutation name before after count expected
        set at 0;set matches 0
        while {[set at [string first $before $original $at]]>=0} {incr matches;incr at}
        if {$matches!=$count} {error "Mutation anchor mismatch: $name"}
        set patched [string map [list $before $after] $original]
        set private [file join $output $name.sv]
        set f [open $private {WRONLY CREAT EXCL}];puts -nonewline $f $patched;close $f
        create_project join [file join $output $name] -part xcvu5p-flva2104-2-i
        add_files $private
        add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_drain_certificate_join.sv]
        set_property top tb_ntrpu_drain_certificate_join [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $output $name join.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first $expected $text]<0 || [string first JOIN_SUITE_PASS $text]>=0} {error "Mutant did not fail intended assertion: $name"}
        puts "JOIN_RTL_MUTANT_REJECTED name=$name"
        close_project
    }
    create_project join_synth [file join $output synthesis] -part xcvu5p-flva2104-2-i
    add_files $source
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_drain_certificate_join [get_filesets sources_1]
    synth_design -mode out_of_context -top ntrpu_drain_certificate_join -part xcvu5p-flva2104-2-i
    report_utilization -file [file join $output join_utilization.rpt]
    report_timing_summary -file [file join $output join_timing.rpt]
    check_timing -verbose -file [file join $output join_check_timing.rpt]
    report_cdc -details -file [file join $output join_cdc.rpt]
    report_exceptions -file [file join $output join_exceptions.rpt]
    set f [open [file join $output join_cells.rpt] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    write_checkpoint [file join $output join_synth.dcp]
    write_verilog -mode funcsim [file join $output join_funcsim.v]
    close_project
    puts NTRPU_DRAIN_JOIN_CHARACTERIZATION_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {characterize_join $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
