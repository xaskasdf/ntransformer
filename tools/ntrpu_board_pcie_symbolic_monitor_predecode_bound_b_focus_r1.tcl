set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc bound_b_focus_case {output mode candidate} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project bound_b_focus [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_numeric_pkg_r1.sv fpga/rtl/ntrpu_pcie_symbolic_delivery_r1.sv} {
        lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]
    }
    lappend sources [file normalize [file join $ntrpu_script_dir .. $candidate]]
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_delivery_predecode_bound_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_delivery_predecode_bound_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_BOUND_B_FOCUS_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project bound_b_focus.sim sim_1 behav xsim simulate.log]
    set stream [open $logfile r];set text [read $stream];close $stream
    file copy $logfile [file join $folder simulate.log]
    if {$mode eq "positive"} {
        if {$caught || [string first "B_BOUND_PASS rows=211606 pre=211606 post=211606 arithmetic_cases=131072 control_cases=65536 bit_cases=10892 random_cases=4096 cold_cases=4 output_bits=2967" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "B bound positive failed: $message"}
    } elseif {[string first "B_BOUND_PASS" $text]>=0 || ![regexp {Fatal: B_BOUND_MISMATCH row=} $text]} {
        error "B bound mutant missed intended public comparator: $mode"
    }
    close_project
    puts "NTRPU_BOUND_B_FOCUS_CASE_REVIEWED $mode"
}
proc bound_b_focus_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    bound_b_focus_case $output positive fpga/rtl/ntrpu_pcie_symbolic_delivery_predecode_bound_r1.sv
    foreach mutation {stale_b rotate_query omit_offset narrow_13 lose_unknown ignore_cold} {
        bound_b_focus_case $output $mutation fpga/tb/ntrpu_pcie_symbolic_delivery_predecode_bound_bad_${mutation}_r1.sv
    }
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_FOCUS_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {bound_b_focus_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
