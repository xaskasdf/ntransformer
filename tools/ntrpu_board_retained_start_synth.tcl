set root [file normalize [file join [file dirname [info script]] ..]]
proc retained_start_synth {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project retained_start [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_reset_supervisor ntrpu_dual_mig_retained_start} {
        add_files [file join $root fpga rtl $module.sv]
    }
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    puts $f {create_clock -name source_clk -period 5.000 [get_ports source_clk]}
    puts $f {create_clock -name ui0_clk -period 5.000 [get_ports {ui_clk[0]}]}
    puts $f {create_clock -name ui1_clk -period 5.000 [get_ports {ui_clk[1]}]}
    close $f
    add_files -fileset constrs_1 [file join $output clocks.xdc]
    set_property top ntrpu_dual_mig_retained_start [get_filesets sources_1]
    update_compile_order -fileset sources_1
    synth_design -mode out_of_context -top ntrpu_dual_mig_retained_start -part xcvu5p-flva2104-2-i
    report_utilization -hierarchical -file [file join $output retained_start_utilization.rpt]
    report_timing_summary -file [file join $output retained_start_timing.rpt]
    check_timing -verbose -file [file join $output retained_start_check_timing.rpt]
    report_cdc -details -file [file join $output retained_start_cdc.rpt]
    report_exceptions -file [file join $output retained_start_exceptions.rpt]
    report_clock_interaction -file [file join $output retained_start_clock_interaction.rpt]
    # Audit the configuration-lifetime state and reset rails in the real netlist.
    # No board I/O delays or blanket asynchronous clock groups are invented.
    report_timing -max_paths 30 -path_type full_clock_expanded -to [get_ports {source_reset_n memory_reset_n[*] peers_reset_req admit restart_required running}] -file [file join $output retained_start_public_outputs.rpt]
    set f [open [file join $output retained_start_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tblackbox\tinit"
    foreach c [get_cells -hierarchical] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
    }
    close $f
    set f [open [file join $output source_files.tsv] {WRONLY CREAT EXCL}]
    puts $f "path\ttype\tused_in_synthesis\tprocessing_order"
    foreach source [get_files -all] {
        puts $f "$source\t[get_property FILE_TYPE $source]\t[get_property USED_IN_SYNTHESIS $source]\t[get_property -quiet PROCESSING_ORDER $source]"
    }
    close $f
    write_checkpoint [file join $output retained_start_synth.dcp]
    write_verilog -mode funcsim [file join $output retained_start_funcsim.v]
    close_project
    puts NTRPU_RETAINED_START_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_start_synth $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
