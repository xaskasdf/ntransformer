set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_synthesis {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project -in_memory -part xcvu5p-flva2104-2-i
    set source [file normalize [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_rc_event_stage_r2.sv]]
    read_verilog -sv [file normalize [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_rc_proposal.sv]]
    read_verilog -sv $source
    # Apply the characterization constraints during synthesis, not only afterwards.
    set xdc [file join $output pcie_rc_event_stage_ooc_input.xdc]
    set f [open $xdc w]
    puts $f {create_clock -name pcie_user_clk -period 4.000 [get_ports clk]}
    puts $f {set_input_delay -clock pcie_user_clk 0.0 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
    puts $f {set_output_delay -clock pcie_user_clk 0.0 [all_outputs]}
    close $f
    read_xdc $xdc
    synth_design -top ntrpu_pcie_rc_event_stage -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
    report_clocks -file [file join $output clocks.rpt]
    report_property -all -file [file join $output source_properties.rpt] [get_files $source]
    report_utilization -file [file join $output utilization.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    report_exceptions -summary -file [file join $output exception_summary.rpt]
    write_checkpoint [file join $output pcie_rc_event_stage.dcp]
    write_verilog -mode funcsim [file join $output pcie_rc_event_stage_funcsim.v]
    write_xdc [file join $output pcie_rc_event_stage.xdc]
    set f [open [file join $output primitives.tsv] w]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    close_project
    puts NTRPU_PCIE_RC_EVENT_STAGE_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_synthesis [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
