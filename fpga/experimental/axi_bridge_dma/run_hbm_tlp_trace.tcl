set out [file normalize $::env(NTRPU_NATIVE_OUT)]
if {[file exists $out]} {error "Fresh output required"}
file mkdir $out
set names {/board/sys_rst_n /board/AXI_PCIE_EP/axi_aclk_out /board/AXI_PCIE_EP/axi_aresetn /board/AXI_PCIE_EP/user_lnk_up /board/RP/user_clk /board/RP/user_lnk_up /board/probe/phase /board/probe/case_id /board/probe/cold_rst_n}
foreach bus {s_axis_rq m_axis_rc} {
    foreach signal {tvalid tready tdata tkeep tlast tuser} {lappend names /board/RP/${bus}_$signal}
}
foreach signal {pcie_rq_tag0 pcie_rq_tag_vld0 pcie_rq_tag1 pcie_rq_tag_vld1} {lappend names /board/RP/pcie_4_0_rport/$signal}
foreach signal {awaddr awlen awsize awvalid awready wdata wstrb wlast wvalid wready bresp bvalid bready araddr arlen arsize arvalid arready rdata rresp rlast rvalid rready} {
    lappend names /board/AXI_PCIE_EP/m_axi_$signal
}
foreach signal {awaddr awvalid awready wdata wstrb wvalid wready bresp bvalid bready araddr arvalid arready rdata rresp rvalid rready} {
    lappend names /board/AXI_PCIE_EP/csr_$signal
}
foreach signal {m_awaddr m_awid m_awlen m_awsize m_awburst m_awvalid m_awready m_wdata m_wstrb m_wlast m_wvalid m_wready m_bvalid m_bready m_bresp m_bid cfg_valid cfg_ready cfg_session cfg_generation cfg_axi_base cfg_pcie_base cfg_rsp_valid cfg_rsp_ready cfg_rsp_error cmd_valid cmd_ready cmd_tag cmd_generation cmd_source cmd_offset cmd_bytes ack_valid ack_ready ack_tag ack_generation ack_bytes result_valid result_ready result_tag result_bytes owned source_owned fault mapper_fault source_fault guard_fault reader_fault_raw availability_fault map_verified hbm_init_complete fabric_clock_locked reader_started_raw client_link_up client_cold_rst_n reader_clk mem_clk mgmt_clk mem_cold_rst_n reader_cold_rst_n management_cold_rst_n} {lappend names /board/AXI_PCIE_EP/$signal}
foreach signal {awaddr awvalid awready wdata wstrb wvalid wready bresp bvalid bready araddr arvalid arready rdata rresp rvalid rready} {lappend names /board/AXI_PCIE_EP/m_axil_$signal}
foreach signal {load_awaddr load_awlen load_awsize load_awvalid load_awready load_wdata load_wstrb load_wlast load_wvalid load_wready load_bresp load_bid load_bvalid load_bready reference_clk reference_enable fabric_reset transfer_id} {lappend names /board/probe/$signal}
foreach signal {h_araddr h_arlen h_arid h_arsize h_arburst h_arvalid h_arready h_rdata h_rresp h_rid h_rlast h_rvalid h_rready} {lappend names /board/AXI_PCIE_EP/$signal}
foreach signal {tvalid tready tdata tkeep tlast tuser} {lappend names /board/RP/m_axis_cq_$signal}
set objects {};set f [open $out/trace_objects.txt {WRONLY CREAT EXCL}]
foreach name $names {
    set found [get_objects $name]
    if {[llength $found]!=1} {error "Missing $name"}
    puts $f $name;lappend objects {*}$found
};close $f
open_vcd $out/native.vcd;log_vcd $objects
if {[info exists ::env(NTRPU_BAR_BASE)]} {
    # Deposit only into the Root Port testbench allocator after its t=0 initial
    # assignment. The existing BAR_INIT task later configures the DUT via PCIe.
    run 1 us
    set allocator /board/RP/tx_usrapp/BAR_INIT_P_MEM32_START
    if {[llength [get_objects $allocator]]!=1} {error "Missing Root Port allocator"}
    set requested $::env(NTRPU_BAR_BASE)
    set_value -radix hex $allocator $requested
    set actual [get_value -radix hex $allocator]
    if {[scan $actual %x actual_number]!=1 || [scan $requested %x requested_number]!=1 || $actual_number!=$requested_number} {
        error "Root Port allocator deposit mismatch: $actual / $requested"
    }
    puts "BT BFM_CONFIG method=ROOT_PORT_ALLOCATOR_DEPOSIT base=$actual"
    flush stdout
}
set previous [current_time -s]
for {set i 0} {$i<301} {incr i} {
    run 1 us
    set now [current_time -s]
    if {$i%10==0} {puts "HBT PROGRESS time=$now init=[get_value -radix hex /board/AXI_PCIE_EP/hbm_init_complete] locked=[get_value /board/AXI_PCIE_EP/fabric_clock_locked] phase=[get_value /board/probe/phase]";flush stdout}
    if {$now eq $previous} {break}
    set previous $now
}
close_vcd
puts HBM_TLP_SIMULATOR_RETURNED
quit
