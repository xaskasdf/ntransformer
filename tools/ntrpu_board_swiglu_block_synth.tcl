# V7 independent Vivado synthesis of HLS-generated SwiGLU/Q8 block for the same VU5P.
# No DDR, PCIe, external delay assumptions or physical timing approval.
proc ntrpu_swiglu_block_synth {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected COMPLETED_HLS_RUN NEW_DIRECTORY"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    set report [file join $input project solution1 syn report ntrpu_swiglu_q8_block_csynth.xml]
    set stream [open $report r]; set xml [read $stream]; close $stream
    if {![string match {*<Part>xcvu5p-flva2104-2-i</Part>*} $xml]} {
        error "HLS report is not for the provisional VU5P"
    }
    set stream [open [file join $input manifest.json] r]; set manifest [read $stream]; close $stream
    if {![string match {*"result": "tool_succeeded"*} $manifest] ||
        ![string match {*"stage": "cosim"*} $manifest]} { error "HLS synthesis not complete" }
    file mkdir $output
    file copy [file join $input project solution1 syn verilog] [file join $output rtl]
    file copy $report [file join $output input_hls_report.xml]
    file copy [file join $input manifest.json] [file join $output input_hls_manifest.json]
    create_project ntrpu_swiglu_block_ooc [file join $output project] -part xcvu5p-flva2104-2-i
    cd [file join $output rtl]
    set rtl [glob *.v]
    set operators [lsort [glob *_ip.tcl]]
    if {[llength $rtl] < 2 || [llength $operators] < 1} { error "Incomplete generated RTL" }
    add_files $rtl
    set data [glob -nocomplain *.dat]
    if {[llength $data]} { add_files $data }
    foreach ip_script $operators { source $ip_script }
    set_property top ntrpu_swiglu_q8_block [get_filesets sources_1]
    update_compile_order -fileset sources_1
    report_ip_status -file [file join $output ip_status.rpt]
    set stream [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $stream {create_clock -name ap_clk -period 5.000 [get_ports ap_clk]}
    close $stream
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_swiglu_q8_block -mode out_of_context -part xcvu5p-flva2104-2-i
    write_checkpoint [file join $output swiglu_block_synth.dcp]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_clocks -file [file join $output clocks.rpt]
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "SwiGLU/Q8 block has unresolved black boxes; not a valid resource budget"
    }
    puts "NTRPU_SWIGLU_BLOCK_VU5P_OOC_SYNTH_EXPORTED; synthesis resources only, not placement/route or integrated validation"
    close_project
}
if {[catch {ntrpu_swiglu_block_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
