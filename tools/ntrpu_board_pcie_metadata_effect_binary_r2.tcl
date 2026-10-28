set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc metadata_effect_binary {output netlist} {
    global ntrpu_script_dir
    if {[file exists $output] || ![file isfile $netlist]} {error "Invalid output/seed"}
    file mkdir $output
    file copy $netlist [file join $output mapped_dut.v]
    set_param general.maxThreads 2
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_metadata_effect_binary.txt]]
    foreach variant {rtl mapped} {
        set folder [file join $output $variant]
        create_project metadata_binary $folder -part xcvu5p-flva2104-2-i
        if {$variant eq "rtl"} {
            add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_metadata_effect_pkg.sv]
            add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_metadata_effect_stage.sv]
        } else {
            add_files [file join $output mapped_dut.v]
        }
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_metadata_effect_binary.sv]
        set_property top tb_ntrpu_pcie_metadata_effect_binary [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set startup [file normalize [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl]]
        set_property xsim.simulate.custom_tcl $startup [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set logfile [file join $folder metadata_binary.sim sim_1 behav xsim simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        file copy $logfile [file join $folder simulate.log]
        if {$caught || [string first "EFFECT_BINARY_PASS rows=12805" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Effect binary $variant failed: $message"}
        close_project
        puts "NTRPU_METADATA_EFFECT_BINARY_REVIEWED $variant"
    }
    puts NTRPU_PCIE_METADATA_EFFECT_BINARY_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {metadata_effect_binary [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
