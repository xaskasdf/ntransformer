set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc axi4_pair_synth {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project pair [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_axi4_pair_drain ntrpu_axi4_drain_monitor ntrpu_tagged_reset_mailbox} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name.sv]
    }
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name host_clk -period 7.000 [get_ports host_clk]}
    puts $f {create_clock -name device_clk -period 5.000 [get_ports device_clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_axi4_pair_drain [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -mode out_of_context -top ntrpu_axi4_pair_drain -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output pair_utilization.rpt]
    report_timing_summary -file [file join $output pair_timing.rpt]
    check_timing -verbose -file [file join $output pair_check_timing.rpt]
    report_cdc -details -file [file join $output pair_cdc.rpt]
    report_exceptions -file [file join $output pair_exceptions.rpt]
    report_clock_interaction -file [file join $output pair_clock_interaction.rpt]
    set f [open [file join $output pair_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    write_checkpoint [file join $output pair_synth.dcp]
    write_verilog -mode funcsim [file join $output pair_funcsim.v]
    close_project
    puts NTRPU_AXI4_PAIR_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_pair_synth [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
