set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc metadata_effect_rtl {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    foreach variant {normal mutant} {
        set folder [file join $output $variant]
        create_project metadata_effect $folder -part xcvu5p-flva2104-2-i
        add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_metadata_effect_pkg.sv]
        set top tb_ntrpu_pcie_metadata_effect
        if {$variant eq "normal"} {
            add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_metadata_effect_stage.sv]
        } else {
            append top _mutant
            add_files [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_metadata_effect_stage_bad_order.sv]
        }
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${top}.sv]
        set_property top $top [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        if {[get_property xsim.simulate.custom_tcl [get_filesets sim_1]] ne $startup} {error "Startup mismatch"}
        update_compile_order -fileset sim_1
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder metadata_effect.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$variant eq "normal"} {
            if {$caught || [string first "EFFECT_PASS checks=17114 unknown_injections=4308" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Effect bank failed: $message"}
        } else {
            if {[string first "EFFECT_PASS" $text]>=0 || ![regexp {Fatal: EFFECT_METADATA_MISMATCH check=0 actual=22008009 expected=2200200c} $text]} {error "Actual order mutant not rejected"}
        }
        close_project
        puts "NTRPU_METADATA_EFFECT_CASE_REVIEWED $variant"
    }
    puts NTRPU_PCIE_METADATA_EFFECT_RTL_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {metadata_effect_rtl [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
