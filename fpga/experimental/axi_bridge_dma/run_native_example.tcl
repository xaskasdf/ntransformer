# Compile and run only a fresh copy of a private AMD bridge example.
set here [file dirname [file normalize [info script]]]
set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {error "SEED_EXAMPLE NEW_OUTPUT required"}
    lassign $argv seed out
    set seed [file normalize $seed];set out [file normalize $out]
    if {[file exists $out] || ![file isfile $seed/ntrpu_bridge_ex.xpr]} {error "Invalid paths"}
    file mkdir $out
    file copy $seed $out/example
    exec python3 -B $here/patch_native_example.py $out/example > $out/patch.log 2>@1
    open_project $out/example/ntrpu_bridge_ex.xpr
    if {[get_property PART [current_project]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first "$seed/" [file normalize $f]]==0} {error "Reference to immutable example: $f"}
    }
    foreach name {ntrpu_axi_bridge_writer ntrpu_axi_bridge_mapper ntrpu_axi_bridge_endpoint_client ntrpu_xdma_local_master ntrpu_axi_bridge_native} {
        add_files -fileset sim_1 $here/$name.sv
    }
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    # Vivado already supplies --mt from its thread setting; do not duplicate it.
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options {-testplusarg TESTNAME=ntrpu_bridge_native}] [get_filesets sim_1]
    report_property -all -file $out/simulation_properties.rpt [get_filesets sim_1]
    update_compile_order -fileset sim_1
    report_compile_order -used_in simulation -file $out/compile_order.rpt
    launch_simulation
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
    close_vcd;close_sim;close_project
    puts BRIDGE_NATIVE_SIMULATOR_RETURNED
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
