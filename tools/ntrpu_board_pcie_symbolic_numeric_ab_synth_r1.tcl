set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc numeric_synthesis {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set source {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_numeric_pair_r1.sv} {
        lappend source [file normalize [file join $ntrpu_script_dir .. fpga rtl $name]]
    }
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    set preflight [file join $output preflight]
    create_project numeric_pair_preflight [file join $preflight project] -part xcvu5p-flva2104-2-i
    add_files $source
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_numeric_pair_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_numeric_pair_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg MODE=pair"] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts NTRPU_NUMERIC_PAIR_PREFLIGHT_START
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $preflight project numeric_pair_preflight.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $preflight simulate.log]
    if {$caught || [string first "NUMERIC_FOCUS_PASS mode=pair steps=266681 " $text]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
        error "Numeric pair preflight failed: $message"
    }
    close_project
    puts NTRPU_NUMERIC_PAIR_PREFLIGHT_REVIEW_READY
    foreach {unit top} {
        apply ntrpu_pcie_symbolic_apply_r1
        delivery ntrpu_pcie_symbolic_delivery_r1
        pair ntrpu_pcie_symbolic_numeric_pair_r1
    } {
        set folder [file join $output $unit]
        file mkdir $folder
        puts "NTRPU_NUMERIC_SYNTH_START $unit"
        create_project -in_memory -part xcvu5p-flva2104-2-i
        read_verilog -sv $source
        # Apply the characterization constraints during synthesis, not only afterwards.
        set xdc [file join $folder pcie_symbolic_numeric_${unit}_ooc_input.xdc]
        set f [open $xdc w]
        puts $f {create_clock -name pcie_user_clk -period 4.000 [get_ports clk]}
        puts $f {set_input_delay -clock pcie_user_clk 0.0 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
        puts $f {set_output_delay -clock pcie_user_clk 0.0 [all_outputs]}
        close $f
        read_xdc $xdc
        help synth_design
        synth_design -directive RuntimeOptimized -top $top -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
        write_checkpoint [file join $folder pcie_symbolic_numeric_${unit}.dcp]
        write_edif [file join $folder pcie_symbolic_numeric_${unit}.edf]
        write_verilog -mode funcsim [file join $folder pcie_symbolic_numeric_${unit}_funcsim.v]
        write_xdc [file join $folder pcie_symbolic_numeric_${unit}.xdc]
        report_clocks -file [file join $folder clocks.rpt]
        foreach source_file [get_files $source] {
            report_property -all -file [file join $folder [file tail $source_file]_properties.rpt] $source_file
        }
        report_utilization -file [file join $folder utilization.rpt]
        report_timing_summary -report_unconstrained -max_paths 20 -file [file join $folder timing.rpt]
        check_timing -verbose -file [file join $folder check_timing.rpt]
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
        puts "NTRPU_NUMERIC_SYNTH_REVIEW_READY $unit"
    }
    puts NTRPU_PCIE_SYMBOLIC_NUMERIC_AB_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {numeric_synthesis [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
