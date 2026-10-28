set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_bank {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project pcie_metadata $output -part xcvu5p-flva2104-2-i
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_tag_metadata.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_tag_metadata.sv]
    set_property top tb_ntrpu_pcie_tag_metadata [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_tag_metadata.txt]]
    if {![file isfile $vectors]} {error "Missing frozen vectors"}
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    set f [open [file join $output pcie_metadata.sim sim_1 behav xsim simulate.log] r]
    set text [read $f];close $f
    if {[string first "META_SUITE_PASS " $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "PCIe inventory regression failed"}
    close_project
    puts NTRPU_PCIE_TAG_METADATA_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_bank [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
