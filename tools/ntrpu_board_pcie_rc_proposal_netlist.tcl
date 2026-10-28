# Binary-domain functional replay of a reviewed synthesized inventory netlist.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc inventory_netlist {input output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set source [file join $input pcie_rc_proposal_funcsim.v]
    if {![file isfile $source]} {error "Missing reviewed functional netlist"}
    file mkdir $output
    file copy $source [file join $output input_funcsim.v]
    set_param general.maxThreads 2
    create_project pcie_rc_proposal_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $output input_funcsim.v]
    set_property top ntrpu_pcie_rc_proposal [get_filesets sources_1]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_rc_proposal.sv]
    set_property top tb_ntrpu_pcie_rc_proposal [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_rc_proposal.txt]]
    if {![file isfile $vectors]} {error "Missing frozen binary vectors"}
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg VECTORS=$vectors"] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    set f [open [file join $output project pcie_rc_proposal_netlist.sim sim_1 behav xsim simulate.log] r]
    set text [read $f];close $f
    if {[string first "RC_PROPOSAL_SUITE_PASS rows=69876" $text]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "PCIe inventory netlist regression failed"}
    close_project
    puts NTRPU_PCIE_RC_PROPOSAL_NETLIST_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {inventory_netlist [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
