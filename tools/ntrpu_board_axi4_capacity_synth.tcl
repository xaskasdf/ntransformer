set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc axi4_drain_synth {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project drain [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_drain_monitor.sv]
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name host_clk -period 7.000 [get_ports clk]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_axi4_drain_monitor [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -generic {WRITE_DEPTH=32} -mode out_of_context -top ntrpu_axi4_drain_monitor -part xcvu5p-flva2104-2-i
    if {[llength [get_ports aw_pending*]]!=6 || [llength [get_ports w_pending*]]!=6 || [llength [get_ports ar_pending*]]!=8} {error "Wrong capacity ports"}
    set f [open [file join $output capacity_parameters.rpt] {WRONLY CREAT EXCL}]
    puts $f "WRITE_DEPTH=32 READ_DEPTH=8 ID_WIDTH=4 AW_PENDING_BITS=6 W_PENDING_BITS=6 AR_PENDING_BITS=8"
    close $f
    report_utilization -hierarchical -file [file join $output drain_utilization.rpt]
    report_timing_summary -file [file join $output drain_timing.rpt]
    check_timing -verbose -file [file join $output drain_check_timing.rpt]
    report_cdc -details -file [file join $output drain_cdc.rpt]
    report_exceptions -file [file join $output drain_exceptions.rpt]
    report_clock_interaction -file [file join $output drain_clock_interaction.rpt]
    set f [open [file join $output drain_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    write_checkpoint [file join $output drain_synth.dcp]
    write_verilog -mode funcsim [file join $output drain_funcsim.v]
    close_project
    puts NTRPU_AXI4_CAPACITY_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_drain_synth [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
