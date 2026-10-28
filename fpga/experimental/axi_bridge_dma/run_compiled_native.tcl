# Reuse an already elaborated snapshot in a private working copy.
# Run with xsim board_behav -tclbatch THIS_FILE -testplusarg TESTNAME=ntrpu_bridge_native.
# NTRPU_NATIVE_OUT must name a NEW report directory.
set out [file normalize $::env(NTRPU_NATIVE_OUT)]
if {[file exists $out]} {error "Expected new report directory"}
file mkdir $out
    set names {/board/sys_rst_n /board/AXI_PCIE_EP/axi_aclk_out /board/AXI_PCIE_EP/axi_aresetn /board/AXI_PCIE_EP/axi_ctl_aresetn /board/AXI_PCIE_EP/user_lnk_up /board/RP/user_clk /board/RP/user_lnk_up}
    foreach suffix {phase case_id cold_rst_n link_up cfg_valid cfg_ready cfg_session cfg_generation cfg_axi_base cfg_pcie_base cfg_rsp_valid cfg_rsp_ready cfg_rsp_error cmd_valid cmd_ready owned fault mapper_fault fault_code mapper_fault_code map_verified ack_valid ack_ready ack_tag ack_generation ack_bytes result_valid result_ready result_tag result_bytes} {
        lappend names /board/probe/$suffix
    }
    foreach suffix {awaddr awprot awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready} {
        lappend names /board/probe/m_axil_$suffix
    }
    foreach suffix {awaddr awid awlen awsize awburst awvalid awready wdata wstrb wlast wvalid wready bvalid bready bresp bid} {
        lappend names /board/probe/m_$suffix
    }
    foreach suffix {tvalid tready tdata tkeep tlast tuser} {lappend names /board/RP/m_axis_cq_$suffix}
    set objects {}
    set f [open $out/trace_objects.txt {WRONLY CREAT EXCL}]
    foreach name $names {
        set found [get_objects $name]
        if {[llength $found]!=1} {error "Missing native signal: $name"}
        puts $f $name;lappend objects {*}$found
    }
    close $f
    open_vcd $out/native.vcd;log_vcd $objects
    set previous [current_time -s]
    for {set i 0} {$i<251} {incr i} {
        run 1 us
        set now [current_time -s]
        if {$i%10==0} {puts "BN PROGRESS time=$now";flush stdout}
        if {$now eq $previous} {break}
        set previous $now
    }
close_vcd
puts BRIDGE_NATIVE_SIMULATOR_RETURNED
quit
