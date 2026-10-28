set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_axil_drain_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project drain $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_axil_drain_monitor ntrpu_endpoint_reset_sequencer} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axil_drain_monitor.sv]
    set_property top tb_ntrpu_axil_drain_monitor [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    puts "NTRPU_DRAIN_FINAL_TIME [current_time -s]"
    close_sim
    set log [file join $output drain.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output drain.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "DRAIN_SUITE_PASS cases=13 native_endpoint=0 cdc=0 scope=axil_only" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "AXI-Lite drain bank failed"}
    set_property top ntrpu_axil_drain_monitor [get_filesets sources_1]
    synth_design -mode out_of_context -top ntrpu_axil_drain_monitor -part xcvu5p-flva2104-2-i
    create_clock -name clk -period 10.000 [get_ports clk]
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    write_checkpoint [file join $output drain_synth.dcp]
    close_project
    puts NTRPU_AXIL_DRAIN_EXPORTED
}
if {[catch {ntrpu_axil_drain_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
