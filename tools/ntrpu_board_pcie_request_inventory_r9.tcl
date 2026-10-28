set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_bank {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project pcie_inventory $output -part xcvu5p-flva2104-2-i
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_request_inventory_r3.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_request_inventory_r2.sv]
    set_property top tb_ntrpu_pcie_request_inventory [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_r5.txt]]
    if {![file isfile $vectors]} {error "Missing frozen vectors"}
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    set f [open [file join $output pcie_inventory.sim sim_1 behav xsim simulate.log] r]
    set text [read $f];close $f
    if {[string first "PCIE_INV_SUITE_PASS " $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "PCIe inventory regression failed"}
    close_project
    puts NTRPU_PCIE_REQUEST_INVENTORY_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_bank [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
