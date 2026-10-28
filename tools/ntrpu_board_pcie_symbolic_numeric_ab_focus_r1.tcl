set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc numeric_ab_focus_r1 {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    foreach mode {positive swap_fields drop_unowned_context omit_origin wrong_fault ignore_enable omit_terminal omit_cold} {
        set folder [file join $output $mode]
        create_project numeric_ab_focus_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
        set sources {}
        foreach name {ntrpu_pcie_metadata_effect_pkg.sv ntrpu_pcie_symbolic_frontend_pkg.sv ntrpu_pcie_symbolic_ced_pkg.sv ntrpu_pcie_symbolic_numeric_pkg_r1.sv} {
            lappend sources [file join $ntrpu_script_dir .. fpga rtl $name]
        }
        foreach unit {apply delivery} {
            set candidate [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_${unit}_r1.sv]
            if {($unit eq "apply" && $mode in {swap_fields drop_unowned_context omit_origin wrong_fault}) ||
                ($unit eq "delivery" && $mode in {ignore_enable omit_terminal omit_cold})} {
                set candidate [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_numeric_ab_r1_bad_${mode}.sv]
            }
            lappend sources $candidate
        }
        add_files $sources
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_numeric_ab_focus_r1.sv]
        set_property top tb_ntrpu_pcie_symbolic_numeric_ab_focus_r1 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property xsim.simulate.xsim.more_options "-testplusarg MODE=$mode" [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_NUMERIC_FOCUS_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project numeric_ab_focus_r1.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$mode eq "positive"} {
            if {$caught || [string first "NUMERIC_FOCUS_PASS mode=positive steps=266681 " $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
                error "Numeric focus failed: $message"
            }
        } elseif {[string first "NUMERIC_FOCUS_PASS" $text]>=0 || ![regexp {Fatal: NUMERIC_FOCUS_[A-Z_]+MISMATCH step=} $text]} {
            error "Numeric focus mutant not rejected: $mode"
        }
        close_project
        puts "NTRPU_NUMERIC_FOCUS_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_NUMERIC_AB_FOCUS_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {numeric_ab_focus_r1 [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
