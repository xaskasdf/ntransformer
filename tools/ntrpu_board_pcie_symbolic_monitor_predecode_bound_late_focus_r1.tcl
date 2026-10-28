set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc fault_leaves_case {output mode mutant} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project bound_late_focus_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv ntrpu_pcie_symbolic_header_guard_r1.sv ntrpu_pcie_symbolic_late_validation_r1.sv ntrpu_pcie_symbolic_late_fault_leaves_winner_r1.sv} {
        lappend sources [file normalize [file join $ntrpu_script_dir .. fpga rtl $name]]
    }
    if {$mutant eq {}} {
        lappend sources [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_late_fault_leaves_bound_r1.sv]
    } else {
        lappend sources [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_late_fault_leaves_bound_bad_${mutant}_r1.sv]
    }
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_late_fault_leaves_bound_focus_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_late_fault_leaves_bound_focus_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_BOUND_LATE_FOCUS_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project bound_late_focus_r1.sim sim_1 behav xsim simulate.log]
    set f [open $logfile r];set text [read $f];close $f
    file copy $logfile [file join $folder simulate.log]
    if {$mutant eq {}} {
        if {$caught || ![regexp {(?m)^BOUND_LATE_PASS samples=309884 bits=16 input_bits=2892 lane_cases=64 unknown_cases=17352 random_cases=20000 zero=[0-9]+ nonzero=[0-9]+ rng=[0-9a-fA-F]+$} $text] ||
            ![regexp {(?m)^BOUND_LATE_PREFIX_CASES count=10240 lane_vectors_per_sample=32 bits_per_lane=16$} $text] ||
            ![regexp {(?m)^BOUND_LATE_BOUND_CASES count=262144 directed=6 lane_vectors_per_sample=32 bits_per_lane=16$} $text] ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
            error "Direct fault-leaf comparison failed: $message"
        }
    } else {
        if {[string first "BOUND_LATE_PASS" $text]>=0 ||
            ![regexp {(?m)^Fatal: BOUND_LATE_(MISMATCH|LANE_MISMATCH) sample=[0-9]+ label=} $text]} {
            error "Fault-leaf mutant missed intended first comparator witness: $mode"
        }
    }
    close_project
    puts "NTRPU_BOUND_LATE_FOCUS_CASE_REVIEWED $mode"
}
proc fault_leaves {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    fault_leaves_case $output positive {}
    foreach mutant {omit_context inclusive_end narrow_13 rotate_prior_query unknown_as_max omit_lower} {
        fault_leaves_case $output $mutant $mutant
    }
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_BOUND_LATE_FOCUS_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {fault_leaves [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
