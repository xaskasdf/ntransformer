# Deliberate faulty RTL copies tested against the frozen real-CDC bank.
# Controlled loss/reset bank; not native PCIe, detector, NTRD ownership or DDR.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_boundary_test {output mutant} {
    global ntrpu_script_dir
    if {[llength [get_projects -quiet]] || [file exists $output]} {error "Expected fresh project and NEW_DIRECTORY"}
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
    set input [file join $ntrpu_script_dir .. fpga rtl ntrpu_retained_channel.sv]
    set f [open $input r];set rtl [read $f];close $f
    if {$mutant eq "erase_requests"} {
        set before {if(!cold_rst_n) begin}
        set after {if(!cold_rst_n || !boundary_rst_n) begin}
    } elseif {$mutant eq "early_response_discard"} {
        set before {TO_BOUNDARY && (fenced || (invalid && !boundary_rst_n))}
        set after {TO_BOUNDARY && (fenced || invalid)}
    } else {error "Unknown mutant"}
    if {[string first $before $rtl]<0 || [string first $before $rtl [expr {[string first $before $rtl]+1}]]>=0} {error "Mutation site is not unique"}
    set modified [string map [list $before $after] $rtl]
    set variant [file join $output ntrpu_retained_channel.sv]
    set f [open $variant {WRONLY CREAT EXCL}];puts -nonewline $f $modified;close $f
    add_files $variant
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_host_boundary_cut.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_host_boundary_cut_r2.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_host_boundary_cdc_wiring.vh]
    set_property file_type {Verilog Header} [get_files *ntrpu_host_boundary_cdc_wiring.vh]
    set_property include_dirs [list [file normalize [file join $ntrpu_script_dir .. fpga tb]]] [get_filesets sim_1]
    set_property top tb_ntrpu_host_boundary_cut_r2 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output boundary.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output boundary.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "BOUND_SUITE_PASS cases=15" $text]<0 && [string first "Fatal:" $text]<0} {error "Mutant simulation did not reach a verdict"}
    puts "NTRPU_BOUNDARY_MUTANT_EXPORTED $mutant"
    close_project
}
if {[catch {
    if {[llength $argv]!=1} {error "Expected NEW_DIRECTORY"}
    set root [file normalize [lindex $argv 0]]
    if {[file exists $root]} {error "Root already exists"}
    file mkdir $root
    foreach mutant {erase_requests early_response_discard} {
        ntrpu_boundary_test [file join $root $mutant] $mutant
    }
} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
