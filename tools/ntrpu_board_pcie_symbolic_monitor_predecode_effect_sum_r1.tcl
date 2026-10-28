set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc effect_sum_case {output mode candidate} {
    global ntrpu_script_dir
    set folder [file join $output $mode]
    create_project effect_sum [file join $folder project] -part xcvu5p-flva2104-2-i
    set sources {}
    foreach name {fpga/rtl/ntrpu_pcie_metadata_effect_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_frontend_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_ced_pkg.sv fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_r1.sv} {
        lappend sources [file normalize [file join $ntrpu_script_dir .. $name]]
    }
    lappend sources [file normalize [file join $ntrpu_script_dir .. $candidate]]
    add_files $sources
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_effects_predecode_sum_r1.sv]
    set_property top tb_ntrpu_pcie_symbolic_effects_predecode_sum_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    puts "NTRPU_EFFECT_SUM_CASE_START $mode"
    launch_simulation
    set caught [catch {run all} message]
    close_sim
    set logfile [file join $folder project effect_sum.sim sim_1 behav xsim simulate.log]
    set stream [open $logfile r];set text [read $stream];close $stream
    file copy $logfile [file join $folder simulate.log]
    if {$mode eq "positive"} {
        if {$caught || [string first "E_SUM_PASS rows=283148 pre=283148 post=283148 control_cases=262144 word_bit_cases=16896 random_cases=4096 cold_cases=4 output_bits=4554" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Effect sum positive failed: $message"}
    } elseif {[string first "E_SUM_PASS" $text]>=0 || ![regexp {Fatal: E_SUM_MISMATCH row=} $text]} {
        error "Effect sum mutant missed intended public comparator: $mode"
    }
    close_project
    puts "NTRPU_EFFECT_SUM_CASE_REVIEWED $mode"
}
proc effect_sum_batch {output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    effect_sum_case $output positive fpga/rtl/ntrpu_pcie_symbolic_effects_predecode_sum_r1.sv
    foreach mutation {cap_2048 omit_append latest_first ignore_unknown apply_from_active unknown_prefix_active} {
        effect_sum_case $output $mutation fpga/tb/ntrpu_pcie_symbolic_effects_predecode_sum_bad_${mutation}_r1.sv
    }
    puts NTRPU_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {effect_sum_batch [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
