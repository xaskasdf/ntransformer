# Guard-only OOC characterization. No timing waiver or board closure.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_guard_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project guard $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_reset_supervisor ntrpu_host_loss_guard} {add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]}
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    foreach {port period} {mgmt_clk 10.000 source_clk 5.000 host_clk 7.300} {
        puts $f "create_clock -name $port -period $period \[get_ports $port\]"
    }
    close $f
    read_xdc [file join $output clocks.xdc]
    set_property top ntrpu_host_loss_guard [get_filesets sources_1]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach source [get_files -compile_order sources -used_in synthesis] {puts $f $source}
    close $f
    synth_design -mode out_of_context -top ntrpu_host_loss_guard -part xcvu5p-flva2104-2-i
    if {[llength [get_clocks]]!=3} {error "Expected three independent parent clocks"}
    foreach {port period} {mgmt_clk 10.000 source_clk 5.000 host_clk 7.300} {
        if {[get_property PERIOD [get_clocks $port]]!=$period} {error "Unexpected clock $port"}
    }
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Blackbox or latch"}
    write_checkpoint [file join $output guard_synth.dcp]
    write_verilog -mode funcsim [file join $output guard_funcsim.v]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    set f [open [file join $output cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tinit\tasync_reg"
    foreach c [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property -quiet INIT $c]\t[get_property -quiet ASYNC_REG $c]"
    }
    close $f
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Host loss guard OOC\n\nManagement10ns, source5ns, host7.3ns assumed independent parent clocks. No external I/O delays, parent clock locations, physical pinout or board clocks. No new false paths, multicycle exceptions or waivers. XPM implementation constraints remain. Guard only: no retained payload queues, AXI Clock Converters, NTRD core, PCIe or DDR. CDC/methodology reports require review; a tool code0 does not qualify reset timing."
    close $f
    puts NTRPU_HOST_LOSS_GUARD_SYNTH_EXPORTED
    close_project
}
if {[catch {ntrpu_guard_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
