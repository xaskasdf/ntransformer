# Isolated synthesis of the previously simulated multibuffer profiles.
set script_dir [file dirname [file normalize [info script]]]
proc multibuffer_synth {arguments} {
    global script_dir
    if {[llength $arguments] != 2} {error "Expected RESERVE_BANKS NEW_DIRECTORY"}
    lassign $arguments reserve output
    if {$reserve ni {4 8}} {error "Only validated reserves 4 and 8"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set part xcvu5p-flva2104-2-i
    set_param general.maxThreads 2
    create_project -in_memory -part $part
    read_verilog -sv [file join $script_dir .. fpga rtl ntrpu_buffer_pool_allocator.sv]
    read_verilog -sv [file join $script_dir .. fpga rtl ntrpu_peer_multibuffer.sv]
    read_xdc [file join $script_dir .. fpga constraints ntrpu_multibuffer_ooc.xdc]
    synth_design -top ntrpu_peer_multibuffer -part $part -mode out_of_context \
        -generic [list CHANNELS=2 RESERVE_BANKS=$reserve BANK_BEATS=64 DESCRIPTORS=16 MAX_OUTSTANDING=4] \
        -flatten_hierarchy rebuilt
    write_xdc -constraints all [file join $output applied.xdc]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_timing -max_paths 10 -file [file join $output critical_paths.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    set latches [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]
    set blackboxes [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]
    set path [get_timing_paths -quiet -delay_type max -max_paths 1]
    set slack [get_property SLACK $path]
    set f [open [file join $output summary.tsv] w]
    foreach {key value} [list part $part reserve_banks $reserve tool_version [version -short] \
        clock_count [llength [get_clocks]] period_ns [get_property PERIOD [get_clocks multibuffer_clk]] \
        latch_count [llength $latches] blackbox_count [llength $blackboxes] \
        worst_setup_slack_ns $slack max_threads [get_param general.maxThreads]] {
        puts $f "$key\t$value"
    }
    close $f
    set f [open [file join $output memories.tsv] w]
    puts $f "cell\tprimitive\tread_width_a\tread_width_b\twrite_width_a\twrite_width_b"
    foreach cell [get_cells -quiet -hier -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM*}] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property READ_WIDTH_A $cell]\t[get_property READ_WIDTH_B $cell]\t[get_property WRITE_WIDTH_A $cell]\t[get_property WRITE_WIDTH_B $cell]"
    }
    close $f
    if {[llength $latches] || [llength $blackboxes] || [llength [get_clocks]] != 1 || $slack eq {}} {
        error "Unexpected internal structure; reports preserved"
    }
    write_checkpoint [file join $output multibuffer_synth.dcp]
    write_verilog -mode funcsim [file join $output multibuffer_funcsim.v]
    puts "MULTIBUFFER_SYNTH_EXPORTED RESERVE=$reserve SETUP_ESTIMATE=$slack"
    puts "SYNTHESIS_ONLY: no placement, routing, external I/O timing or CDC qualification."
    close_project
}
if {[catch {multibuffer_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
