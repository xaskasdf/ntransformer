# Separate side-conversation synthesis; does not use the principal's queue.
set script_dir [file dirname [file normalize [info script]]]
proc peer_synth {arguments} {
    global script_dir
    if {[llength $arguments] != 2} {error "Expected MAX_BEATS NEW_DIRECTORY"}
    lassign $arguments beats output
    if {$beats ni {4 64}} {error "Only validated buffer profiles 4 and 64"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    set part xcvu5p-flva2104-2-i
    set_param general.maxThreads 2
    create_project -in_memory -part $part
    read_verilog -sv [file join $script_dir .. fpga rtl ntrpu_peer_rx_guard.sv]
    read_xdc [file join $script_dir .. fpga constraints ntrpu_peer_rx_ooc.xdc]
    synth_design -top ntrpu_peer_rx_guard -part $part -mode out_of_context \
        -generic MAX_BEATS=$beats -flatten_hierarchy rebuilt
    write_checkpoint [file join $output peer_rx_synth.dcp]
    write_verilog -mode funcsim [file join $output peer_rx_funcsim.v]
    write_xdc -constraints all [file join $output applied.xdc]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_timing -max_paths 5 -file [file join $output critical_paths.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]
    set blackboxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
    set clocks [get_clocks -quiet]
    set path [get_timing_paths -quiet -delay_type max -max_paths 1]
    set slack [get_property SLACK $path]
    set f [open [file join $output summary.tsv] w]
    foreach {key value} [list part $part max_beats $beats tool_version [version -short] \
        max_threads [get_param general.maxThreads] clock_count [llength $clocks] \
        period_ns [get_property PERIOD [get_clocks peer_clk]] \
        latch_count [llength $latches] blackbox_count [llength $blackboxes] \
        worst_setup_slack_ns $slack cell_count [llength [get_cells -hierarchical]]] {
        puts $f "$key\t$value"
    }
    close $f
    if {[llength $clocks] != 1 || [llength $latches] != 0 || [llength $blackboxes] != 0} {
        error "Unexpected clock/latch/blackbox structure; reports preserved"
    }
    if {$slack eq {}} {error "No constrained internal setup path"}
    puts "NTRPU_PEER_RX_SYNTH_EXPORTED MAX_BEATS=$beats SETUP_ESTIMATE=$slack"
    puts "SYNTHESIS_ONLY: external I/O timing, placement, routing and CDC are not qualified."
    close_project
}
if {[catch {peer_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
