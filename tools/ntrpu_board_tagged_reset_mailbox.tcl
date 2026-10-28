set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_mailbox_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project mailbox $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_tagged_reset_mailbox ntrpu_axil_drain_monitor ntrpu_endpoint_reset_sequencer} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_tagged_reset_mailbox.sv]
    set_property top tb_ntrpu_tagged_reset_mailbox [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    puts "NTRPU_MAILBOX_FINAL_TIME [current_time -s]"
    close_sim
    set log [file join $output mailbox.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output mailbox.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "MAILBOX_SUITE_PASS cases=12 deliveries=13 answers=10 obsolete=3" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Tagged mailbox bank failed"}
    set clocks [file join $output clocks.xdc]
    set f [open $clocks {WRONLY CREAT EXCL}]
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    puts $f {create_clock -name peer_clk -period 7.000 [get_ports peer_clk]}
    close $f
    add_files -fileset constrs_1 $clocks
    set_property top ntrpu_tagged_reset_mailbox [get_filesets sources_1]
    synth_design -mode out_of_context -top ntrpu_tagged_reset_mailbox -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    write_checkpoint [file join $output mailbox_synth.dcp]
    close_project
    puts NTRPU_TAGGED_MAILBOX_EXPORTED
}
if {[catch {ntrpu_mailbox_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
