# Source-only OOC synthesis, corrected top port; prior diagnostic bank preserved.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_retained_faults {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project retained [file join $output synthesis] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_retained_channel ntrpu_host_boundary_cut} {add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]}
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name source_clk -period 5.000 [get_ports clk]}
    close $f
    read_xdc [file join $output clocks.xdc]
    set_property top ntrpu_host_boundary_cut [get_filesets sources_1]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach source [get_files -compile_order sources -used_in synthesis] {puts $f $source}
    close $f
    synth_design -mode out_of_context -top ntrpu_host_boundary_cut -part xcvu5p-flva2104-2-i
    if {[llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks source_clk]]!=5.0} {error "Wrong source clock"}
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Blackbox or latch"}
    write_checkpoint [file join $output retained_synth.dcp]
    write_verilog -mode funcsim [file join $output retained_funcsim.v]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Retained source boundary OOC\n\nEight two-entry channels with 64-bit counters. Single assumed 5 ns source clock. No external I/O delays, parent clock locations or board pins. No timing exceptions or waivers added. No CDC, detector, PCIe, DDR or NTRD core. Synthesis resource estimate only; no placement/routing or physical timing qualification. The diagnostic bank separately uses 2-bit counters to exercise saturation with real traffic."
    close $f
    puts NTRPU_RETAINED_SYNTH_EXPORTED
    close_project
}
if {[catch {ntrpu_retained_faults $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
