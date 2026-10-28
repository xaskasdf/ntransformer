# V7 synthesis budget for the complete validated transport + MAC job controller.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_gemv_stream_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] ni {2 3} || [llength [get_projects -quiet]]} { error "Expected HLS_MAC_RUN NEW_DIRECTORY ?scalar|dual?" }
    lassign $arguments input output variant
    if {$variant eq ""} { set variant scalar }
    if {$variant ni {scalar dual}} { error "Unknown GEMV variant" }
    set top [expr {$variant eq "dual" ? "ntrpu_q8_gemv_parallel" : "ntrpu_q8_gemv_stream"}]
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    set stream [open [file join $input manifest.json] r]; set manifest [read $stream]; close $stream
    foreach pattern {{*"result": "tool_succeeded"*} {*"kernel": "q8_block_mac"*} {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}} {
        if {![string match $pattern $manifest]} { error "MAC manifest mismatch: $pattern" }
    }
    file mkdir $output
    set stream [open [file join $output variant.txt] {WRONLY CREAT EXCL}]
    puts $stream $variant
    close $stream
    file copy [file join $input project solution1 syn verilog] [file join $output mac_rtl]
    file copy [file join $input manifest.json] [file join $output input_hls_manifest.json]
    create_project ntrpu_gemv_stream_synth [file join $output project] -part xcvu5p-flva2104-2-i
    cd [file join $output mac_rtl]
    add_files [glob *.v]
    foreach script [lsort [glob *_ip.tcl]] { source $script }
    foreach name [list ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream $top] {
        add_files [file normalize [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]]
    }
    set_property top $top [get_filesets sources_1]
    update_compile_order -fileset sources_1
    report_ip_status -file [file join $output ip_status.rpt]
    set stream [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $stream {create_clock -name clk -period 5.000 [get_ports clk]}
    close $stream
    read_xdc [file join $output clock.xdc]
    synth_design -top $top -mode out_of_context -part xcvu5p-flva2104-2-i
    write_checkpoint [file join $output gemv_stream_synth.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_clocks -file [file join $output clocks.rpt]
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
    puts "NTRPU_GEMV_STREAM_SYNTH_EXPORTED; mapped budget only, not placed or routed"
    close_project
}
if {[catch {ntrpu_gemv_stream_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
