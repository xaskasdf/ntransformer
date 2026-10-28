# Retained interface boundary with real AMD AXI4/AXI-Lite CDCs.
# Automatic guard-driven resets; controlled owner/peer ACK, not native PCIe/NTRD/DDR.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_boundary_test {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project boundary $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach {name protocol width address} {ntrpu_ingress_axi_cdc AXI4 512 64 ntrpu_host_axil_cdc AXI4LITE 32 32} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property -dict [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $ip
        if {$protocol eq "AXI4"} {set_property CONFIG.ID_WIDTH 4 $ip}
        generate_target all $ip
        report_property -all $ip -file [file join $output ${name}_configured.rpt]
    }
    foreach name {ntrpu_retained_channel ntrpu_host_boundary_cut ntrpu_reset_supervisor ntrpu_host_loss_guard} {add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]}
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_host_guarded_boundary.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_host_boundary_cdc_wiring.vh]
    set_property file_type {Verilog Header} [get_files *ntrpu_host_boundary_cdc_wiring.vh]
    set_property include_dirs [list [file normalize [file join $ntrpu_script_dir .. fpga tb]]] [get_filesets sim_1]
    set_property top tb_ntrpu_host_guarded_boundary [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output boundary.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output boundary.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "BOUND_SUITE_PASS cases=17" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Boundary bank failed"}
    puts NTRPU_HOST_GUARDED_BOUNDARY_EXPORTED
    close_project
}
if {[catch {ntrpu_boundary_test $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
