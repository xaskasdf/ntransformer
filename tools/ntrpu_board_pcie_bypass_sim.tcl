# Actual BAR MemWr/MemRd, bounded behavioral sink; no DDR or P2P claim.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_bypass_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {
        error "Use fresh process: INPUT_EXAMPLE NEW_OUTPUT"
    }
    lassign $arguments input output
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output] || ![file exists [file join $input ntrpu_xdma_ex.xpr]]} {
        error "Missing input or output already exists"
    }
    file mkdir $output
    file copy $input [file join $output example]
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_bypass.py] [file join $output example] > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first $input [file normalize $f]]==0} {error "Copied project references immutable input: $f"}
    }
    set ip [get_ips ntrpu_xdma]
    foreach {key value} {CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axi_bypass_64bit_en true CONFIG.pciebar2axibar_axist_bypass 0x0000000100000000} {
        if {[get_property $key $ip] ne $value} {error "Wrong $key"}
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_bypass_memory.sv]
    # Instrumented public root sources use SystemVerilog time literals.
    set_property file_type SystemVerilog [get_files *usp_pci_exp_usrapp_com.v]
    set_property file_type SystemVerilog [get_files *usp_pci_exp_usrapp_tx.v]
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options {-testplusarg TESTNAME=ntrpu_bypass}] [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set previous [current_time -s]
    for {set i 0} {$i<205} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_BYPASS_PROGRESS time=$now"
        flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    close_sim
    close_project
    puts NTRPU_BYPASS_SIMULATOR_RETURNED
}
if {[catch {ntrpu_bypass_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
