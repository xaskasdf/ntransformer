set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc row_enable_case {output mode e_candidate d_candidate} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project row_enable [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_r1.sv} {
        lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]
    }
    lappend sources [file normalize [file join $ntrpu_script_dir .. $e_candidate]]
    lappend sources [file normalize [file join $ntrpu_script_dir .. $d_candidate]]
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_row_enable_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_row_enable_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_ROW_ENABLE_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project row_enable.sim sim_1 behav xsim simulate.log]
    set stream [open $logfile r];set text [read $stream];close $stream
    file copy $logfile [file join $folder simulate.log]
    if {$mode eq "positive"} {
        if {$caught || [string first "ROW_ENABLE_PASS rows=40958 pre=40958 post=40958 binary_cases=16384 four_state_cases=16384 bit_cases=7148 random_cases=1024 table_bits=14336 retained_bits=51 d_output_bits=2498 e_output_bits=4554 enable_bits=512" $text]<0 ||
            [regexp -nocase {(^|
)[ 	]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Row enable positive failed: $message"}
    } elseif {[string first "ROW_ENABLE_PASS" $text]>=0 || ![regexp {Fatal: ROW_ENABLE_MISMATCH row=} $text]} {
        error "Row enable mutant missed intended comparator: $mode"
    }
    close_project
    puts "NTRPU_ROW_ENABLE_CASE_REVIEWED $mode"
}
proc row_enable_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    row_enable_case $output positive fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1.sv
    row_enable_case $output omit_commit fpga/tb/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1_bad_omit_commit.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1.sv
    row_enable_case $output omit_valid fpga/tb/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1_bad_omit_valid.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1.sv
    row_enable_case $output omit_observing fpga/tb/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1_bad_omit_observing.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1.sv
    row_enable_case $output rotate_row fpga/tb/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1_bad_rotate_row.sv fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1.sv
    row_enable_case $output reverse_admissions fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1.sv fpga/tb/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1_bad_reverse_admissions.sv
    row_enable_case $output ignore_cold fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_row_enable_r1.sv fpga/tb/ntrpu_pcie_symbolic_descriptions_predecode_row_enable_r1_bad_ignore_cold.sv
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {row_enable_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
