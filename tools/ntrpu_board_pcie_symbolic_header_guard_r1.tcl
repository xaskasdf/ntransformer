set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc header_guard_r1 {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
    foreach mode {positive ignore_enable omit_terminal signed_compare count_upper zero_extend} {
        set folder [file join $output $mode]
        create_project header_guard_r1 [file join $folder project] -part xcvu5p-flva2104-2-i
        set candidate [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_symbolic_header_guard_r1.sv]
        if {$mode ne "positive"} {set candidate [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_symbolic_header_guard_r1_bad_${mode}.sv]}
        add_files $candidate
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_symbolic_header_guard_r1.sv]
        set_property top tb_ntrpu_pcie_symbolic_header_guard_r1 [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg MODE=$mode"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        puts "NTRPU_HEADER_GUARD_START $mode"
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder project header_guard_r1.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$mode eq "positive"} {
            if {$caught || [string first "HEADER_GUARD_PASS mode=positive steps=1085445 " $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {
                error "Header guard failed: $message"
            }
        } elseif {[string first "HEADER_GUARD_PASS" $text]>=0 || ![regexp {Fatal: HEADER_GUARD_MISMATCH step=} $text]} {
            error "Header guard mutant not rejected: $mode"
        }
        close_project
        puts "NTRPU_HEADER_GUARD_REVIEWED $mode"
    }
    puts NTRPU_SYMBOLIC_HEADER_GUARD_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {header_guard_r1 [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
