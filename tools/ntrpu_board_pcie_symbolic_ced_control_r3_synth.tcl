set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ced_synthesis {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach {unit top} {
        control ntrpu_pcie_symbolic_control_r3
        combined ntrpu_pcie_symbolic_frontend_r3
    } {
        set folder [file join $output $unit]
        file mkdir $folder
        puts "NTRPU_CED_SYNTH_START $unit"
        create_project -in_memory -part xcvu5p-flva2104-2-i
        set source {}
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_r2.sv ntrpu_pcie_symbolic_descriptions_r2.sv ntrpu_pcie_symbolic_frontend_r3.sv} {
            lappend source [file normalize [file join $ntrpu_script_dir .. fpga rtl $name]]
        }
        read_verilog -sv $source
        # Apply the characterization constraints during synthesis, not only afterwards.
        set xdc [file join $folder pcie_symbolic_ced_${unit}_ooc_input.xdc]
        set f [open $xdc w]
        puts $f {create_clock -name pcie_user_clk -period 4.000 [get_ports clk]}
        puts $f {set_input_delay -clock pcie_user_clk 0.0 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
        puts $f {set_output_delay -clock pcie_user_clk 0.0 [all_outputs]}
        close $f
        read_xdc $xdc
        help synth_design
        synth_design -directive RuntimeOptimized -top $top -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
        write_checkpoint [file join $folder pcie_symbolic_ced_${unit}.dcp]
        write_edif [file join $folder pcie_symbolic_ced_${unit}.edf]
        write_verilog -mode funcsim [file join $folder pcie_symbolic_ced_${unit}_funcsim.v]
        write_xdc [file join $folder pcie_symbolic_ced_${unit}.xdc]
        report_clocks -file [file join $folder clocks.rpt]
        foreach source_file [get_files $source] {
            report_property -all -file [file join $folder [file tail $source_file]_properties.rpt] $source_file
        }
        report_utilization -file [file join $folder utilization.rpt]
        report_timing_summary -report_unconstrained -max_paths 20 -file [file join $folder timing.rpt]
        report_drc -file [file join $folder drc.rpt]
        report_methodology -file [file join $folder methodology.rpt]
        report_exceptions -file [file join $folder exceptions.rpt]
        report_exceptions -ignored -file [file join $folder ignored_exceptions.rpt]
        report_exceptions -summary -file [file join $folder exception_summary.rpt]
        set f [open [file join $folder primitives.tsv] w]
        puts $f "cell\tprimitive"
        foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
            puts $f "$cell\t[get_property REF_NAME $cell]"
        }
        close $f
        close_project
        puts "NTRPU_CED_SYNTH_REVIEW_READY $unit"
    }
    puts NTRPU_PCIE_SYMBOLIC_CED_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {ced_synthesis [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
