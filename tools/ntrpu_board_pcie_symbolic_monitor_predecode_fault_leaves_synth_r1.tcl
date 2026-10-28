set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc fault_leaves_case {output mode vector rows mutant directed input_mutant} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    set vectors [file normalize [file join $ntrpu_script_dir .. $vector]]
    create_project predecode_fault_leaves_monitor_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_numeric_pkg_r1.sv fpga/rtl/ntrpu_pcie_rc_update_select_r3.sv fpga/rtl/ntrpu_pcie_request_inventory_r7.sv fpga/rtl/ntrpu_pcie_symbolic_apply_r1.sv fpga/rtl/ntrpu_pcie_symbolic_control_r3.sv fpga/rtl/ntrpu_pcie_symbolic_delivery_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_r2.sv fpga/rtl/ntrpu_pcie_symbolic_effects_r2.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_r1.sv fpga/rtl/ntrpu_pcie_symbolic_header_guard_r1.sv fpga/rtl/ntrpu_pcie_symbolic_late_validation_r1.sv fpga/rtl/ntrpu_pcie_symbolic_monitor_r1.sv fpga/rtl/ntrpu_pcie_symbolic_retirement_r1.sv fpga/tb/ntrpu_pcie_inventory_symbolic_model_r1.sv fpga/tb/ntrpu_pcie_request_inventory_symbolic_late_probe_r1.sv fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_late_fault_leaves_r1.sv fpga/rtl/ntrpu_pcie_symbolic_retirement_fault_leaves_r1.sv} {lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]}
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1_bad_${mutant}.sv]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    set options "-testplusarg VECTORS=$vectors -testplusarg BANK=$mode -testplusarg ROWS=$rows"
    if {$directed} {append options " -testplusarg QUIET_DIRECTED"}
    if {$input_mutant ne {}} {append options " -testplusarg $input_mutant"}
    set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project predecode_fault_leaves_monitor_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {} && $input_mutant eq {}} {
        if {$caught || [string first "MONITOR_CANDIDATE_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "MONITOR_PASS bank=$mode rows=$rows pre=[expr {$rows+5}] post=[expr {$rows+5}] " $text]<0 ||
            [string first "RETIREMENT_PASS bank=$mode rows=$rows samples=[expr {$rows+5}] " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Integrated fault-leaf comparison failed: $mode $message"}
    } elseif {[string first "MONITOR_CANDIDATE_PASS" $text]>=0 || ![regexp {Fatal: MONITOR_CANDIDATE_PORT_MISMATCH row=} $text]} {
        error "Integrated fault-leaf control missed intended public comparator: $mode"
    }
    close_project
    puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_CASE_REVIEWED $mode"
}


proc fault_leaves_synthesis {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    fault_leaves_case $output monitor fpga/tb/data/pcie_symbolic_monitor_directed_r2.txt 252 {} 1 {}
    fault_leaves_case $output range_return fpga/tb/data/pcie_symbolic_retirement_range_r1.txt 277 {} 0 {}
    puts NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_PREFLIGHT_REVIEW_READY
    set source {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_rc_update_select_r3.sv ntrpu_pcie_symbolic_control_r3.sv ntrpu_pcie_symbolic_effects_predecode_r1.sv ntrpu_pcie_symbolic_descriptions_predecode_r1.sv ntrpu_pcie_symbolic_apply_r1.sv ntrpu_pcie_symbolic_delivery_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv ntrpu_pcie_symbolic_late_fault_leaves_r1.sv ntrpu_pcie_symbolic_retirement_fault_leaves_r1.sv ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1.sv} {
        lappend source [file normalize [file join $ntrpu_script_dir .. fpga rtl $name]]
    }
    set unit monitor
    set top ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_r1
        set folder [file join $output $unit]
        file mkdir $folder
        puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_SYNTH_START $unit"
        create_project -in_memory -part xcvu5p-flva2104-2-i
        read_verilog -sv $source
        # Apply the characterization constraints during synthesis, not only afterwards.
        set xdc [file join $folder pcie_symbolic_monitor_predecode_fault_leaves_${unit}_ooc_input.xdc]
        set f [open $xdc w]
        puts $f {create_clock -name pcie_user_clk -period 4.000 [get_ports clk]}
        puts $f {set_input_delay -clock pcie_user_clk 0.0 [get_ports -filter {DIRECTION == IN && NAME != clk}]}
        puts $f {set_output_delay -clock pcie_user_clk 0.0 [all_outputs]}
        close $f
        read_xdc $xdc
        help synth_design
        synth_design -directive RuntimeOptimized -top $top -part xcvu5p-flva2104-2-i -mode out_of_context -flatten_hierarchy rebuilt
        write_checkpoint [file join $folder pcie_symbolic_monitor_predecode_fault_leaves_${unit}.dcp]
        write_edif [file join $folder pcie_symbolic_monitor_predecode_fault_leaves_${unit}.edf]
        write_verilog -mode funcsim [file join $folder pcie_symbolic_monitor_predecode_fault_leaves_${unit}_funcsim.v]
        write_xdc [file join $folder pcie_symbolic_monitor_predecode_fault_leaves_${unit}.xdc]
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
        puts "NTRPU_MONITOR_PREDECODE_FAULT_LEAVES_SYNTH_REVIEW_READY $unit"
    puts NTRPU_PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_SYNTH_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {fault_leaves_synthesis [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
