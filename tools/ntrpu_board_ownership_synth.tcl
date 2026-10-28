set root [file normalize [file join [file dirname [info script]] ..]]
proc ownership_synth {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project ownership [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_pool ntrpu_tagged_reset_mailbox ntrpu_ownership_certificate ntrpu_pcie_fragment_certified_owner} {
        add_files [file join $root fpga rtl $module.sv]
    }
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name source_clk -period 5.000 [get_ports clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_pcie_fragment_certified_owner [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -mode out_of_context -top ntrpu_pcie_fragment_certified_owner -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output ownership_utilization.rpt]
    report_timing_summary -file [file join $output ownership_timing.rpt]
    check_timing -verbose -file [file join $output ownership_check_timing.rpt]
    report_cdc -details -file [file join $output ownership_cdc.rpt]
    report_exceptions -file [file join $output ownership_exceptions.rpt]
    report_clock_interaction -file [file join $output ownership_clock_interaction.rpt]
    set f [open [file join $output ownership_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    write_checkpoint [file join $output ownership_synth.dcp]
    write_verilog -mode funcsim [file join $output ownership_funcsim.v]
    close_project
    puts NTRPU_OWNERSHIP_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {ownership_synth $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
