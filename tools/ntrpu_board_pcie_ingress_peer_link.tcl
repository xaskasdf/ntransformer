# Actual XDMA BAR MemWr -> ingress RTL; destination/isolation BFMs, no DDR/P2P.
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
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_ingress_peer.py] [file join $output example] > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first $input [file normalize $f]]==0} {error "Copied project references immutable input: $f"}
    }
    set ip [get_ips ntrpu_xdma]
    foreach {key value} {CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axi_bypass_64bit_en true CONFIG.pciebar2axibar_axist_bypass 0x0000000100000000} {
        if {[get_property $key $ip] ne $value} {error "Wrong $key"}
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_ingress_peer_harness.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_dword_decode.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_line_ingress.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_peer_rx_guard.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_ingress_peer.sv]
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
set names {/board/RP/user_clk /board/EP/user_clk /board/EP/user_resetn}
foreach prefix {s_axis_rq m_axis_rc} {
    foreach suffix {tdata tkeep tuser tvalid tready tlast} {lappend names /board/RP/${prefix}_${suffix}}
}
foreach name {pcie_rq_tag0 pcie_rq_tag_vld0 pcie_rq_tag1 pcie_rq_tag_vld1} {
    lappend names /board/RP/pcie_4_0_rport/$name
}
foreach suffix {awid awaddr awlen awsize awburst awvalid awready wdata wstrb wlast wvalid wready bid bresp bvalid bready arid araddr arlen arsize arburst arvalid arready rid rdata rresp rlast rvalid rready} {
    lappend names /board/EP/m_axib_$suffix
}
lappend names /board/EP/m_axib_awlock
foreach suffix {stop source_stopped ingress_isolated quiesce_epoch arm_valid arm_ready arm_epoch arm_tag arm_window arm_destination epoch lease_held failed error_code transport_idle valid_bytes line_valid line_ready line_epoch line_tag line_address line_data done_valid done_ready done_epoch done_tag done_error result_valid result_ready result_epoch result_tag result_status} {
    lappend names /board/ntrpu_bypass/$suffix
}
foreach suffix {peer_epoch peer_quarantined peer_axi_busy bridge_phase destination_status m_axi_awaddr m_axi_awid m_axi_awlen m_axi_awsize m_axi_awburst m_axi_awcache m_axi_awprot m_axi_awlock m_axi_awvalid m_axi_awready m_axi_wdata m_axi_wstrb m_axi_wlast m_axi_wvalid m_axi_wready m_axi_bid m_axi_bresp m_axi_bvalid m_axi_bready} {
    lappend names /board/ntrpu_bypass/$suffix
}
set objects {}
set f [open [file join $output trace_objects.txt] {WRONLY CREAT EXCL}]
foreach name $names {
    set found [get_objects $name]
    if {[llength $found]!=1} {error "Expected one native signal: $name"}
    puts $f $name
    lappend objects {*}$found
}
close $f
open_vcd [file join $output native.vcd]
log_vcd $objects

    set previous [current_time -s]
    for {set i 0} {$i<205} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_BYPASS_PROGRESS time=$now"
        flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    close_vcd
    close_sim
    close_project
    puts NTRPU_PEER_LINK_SIMULATOR_RETURNED
}
if {[catch {ntrpu_bypass_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
