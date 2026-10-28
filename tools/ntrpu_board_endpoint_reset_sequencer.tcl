# Isolated sequencer plus actual XPM clock probe; no native XDMA/MIG instance.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_sequence_bank {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project sequence $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_reset_supervisor ntrpu_endpoint_reset_sequencer} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_endpoint_reset_sequencer.sv]
    set_property top tb_ntrpu_endpoint_reset_sequencer [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    puts "NTRPU_SEQUENCE_FINAL_TIME [current_time -s]"
    close_sim
    set log [file join $output sequence.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output sequence.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "SEQ_SUITE_PASS cases=9 full_cycles=5000001" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Endpoint sequencer bank failed"}
    set_property top ntrpu_endpoint_reset_sequencer [get_filesets sources_1]
    synth_design -mode out_of_context -top ntrpu_endpoint_reset_sequencer -part xcvu5p-flva2104-2-i
    create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    write_checkpoint [file join $output sequencer_synth.dcp]
    set f [open [file join $output cell_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "ref_name\tcount"
    foreach ref [lsort -unique [get_property REF_NAME [get_cells -hierarchical]]] {
        puts $f "$ref\t[llength [get_cells -hierarchical -filter [format {REF_NAME == %s} $ref]]]"
    }
    close $f
    close_project
    puts NTRPU_ENDPOINT_SEQUENCE_EXPORTED
}
if {[catch {ntrpu_sequence_bank $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
