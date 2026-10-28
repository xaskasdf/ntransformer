# Private native PIPE gate; no implementation or edits to the source example.
set root [file dirname [file dirname [file normalize [info script]]]]
set out [file normalize [lindex $argv 0]]
set_param general.maxThreads 2
if {[catch {
    file mkdir $out
    set f [open $root/evidence/seed.txt];set seed [string trim [read $f]];close $f
    set example [file join [file dirname $out] work example]
    file mkdir [file dirname $example]
    if {[file exists $root/evidence/cache.txt]} {
        if {![file exists $example]} {error "Runner did not copy the private cache"}
    } else {
        if {[file exists $example]} {error "Expected fresh example"}
        file copy $seed $example
        exec python3 -B $root/tools/patch_ntrpu_hbm_xdma_guarded.py $example
    }
    open_project $example/ntrpu_vu47p_xdma_ex.xpr
    # The private cached project references its frozen old fixture. Replace
    # those file references only; the donor project and its files stay untouched.
    if {[file exists $root/evidence/cache.txt]} {
        foreach f [get_files -all] {
            if {[string first /repo/fpga/ [file normalize $f]]>=0} {remove_files $f}
        }
    }
    if {[get_property PART [current_project]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong part"}
    foreach {key value} {CONFIG.pipe_sim true CONFIG.functional_mode DMA CONFIG.xdma_axi_intf_mm AXI_Stream CONFIG.axi_data_width 512_bit CONFIG.axi_addr_width 64 CONFIG.xdma_dsc_bypass true CONFIG.dsc_bypass_wr 0011 CONFIG.xdma_wnum_chnl 2 CONFIG.xdma_sts_ports true CONFIG.axisten_freq 250 CONFIG.soft_reset_en false} {
        if {[get_property $key [get_ips ntrpu_vu47p_xdma]] ne $value} {error "Wrong $key"}
    }
    foreach f [get_files -all] {
        if {[string first $seed [file normalize $f]]==0} {error "Reference to seed: $f"}
    }
    foreach name {ntrpu_peer_read_pkg ntrpu_bridge_async_fifo ntrpu_hbm_dma_bridge_lane ntrpu_hbm_dma_bridge} {
        add_files -fileset sim_1 $root/fpga/experimental/$name.sv
    }
    add_files -fileset sim_1 $root/fpga/rtl/ntrpu_xdma_local_master.sv
    add_files -fileset sim_1 $root/fpga/experimental/ntrpu_xdma_completion_guard.sv
    add_files -fileset sim_1 $root/fpga/experimental/ntrpu_hbm_xdma_guarded.sv
    add_files -fileset sim_1 $root/fpga/tb/ntrpu_hbm_xdma_guarded_native.sv
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options {-testplusarg TESTNAME=ntrpu_hbm_xdma}] [get_filesets sim_1]
    report_property -all -file $out/xdma_properties.rpt [get_ips ntrpu_vu47p_xdma]
    report_property -all -file $out/simulation_properties.rpt [get_filesets sim_1]
    update_compile_order -fileset sim_1
    report_compile_order -used_in simulation -file $out/compile_order.rpt
    launch_simulation
    set names {/board/EP/user_clk /board/EP/user_resetn /board/EP/user_lnk_up /board/probe/phase /board/probe/native_status /board/probe/dsc_ready /board/probe/dsc_load /board/probe/dsc_src /board/probe/dsc_dst /board/probe/dsc_len /board/probe/dsc_ctl /board/probe/axis_ready /board/probe/axis_valid /board/probe/axis_data /board/probe/axis_last /board/probe/axis_keep /board/probe/dma_done /board/probe/dma_error /board/probe/contexts_held /board/probe/result_valid /board/probe/result_ready /board/probe/ack_valid /board/probe/local_valid}
    lappend names /board/RP/user_clk /board/RP/tx_usrapp/P_READ_DATA
    foreach suffix {clk result_context result_status result_bytes submit_valid submit_ready submit_channel submit_context} {lappend names /board/probe/$suffix}
    foreach suffix {tvalid tready tdata tkeep tlast tuser} {lappend names /board/RP/m_axis_cq_$suffix}
    foreach suffix {guard_faults guard_active guard_start guard_start_ready guard_epoch admit_descriptor validated_done published_count observed_count delivered_count control_snapshot status_snapshot count_snapshot init_arready init_rvalid init_ar_count init_r_count management_core_ready exclusive_owner management_fault management_wait_expired management_reset_request} {lappend names /board/probe/$suffix}
    foreach suffix {m_admit m_req_valid m_req_ready m_req_write m_req_address m_req_data m_req_strobe m_req_identity m_rsp_valid m_rsp_ready m_rsp_data m_rsp_code m_rsp_identity m_rsp_timeout m_timeout_latched} {lappend names /board/probe/dut/$suffix}
    foreach suffix {awaddr awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready} {
        lappend names /board/probe/m_axil_$suffix /board/EP/m_axil_$suffix
    }
    set objects {}
    foreach name $names {
        set found [get_objects $name];if {[llength $found]!=1} {error "Missing $name"};lappend objects {*}$found
    }
    open_vcd $out/native.vcd;log_vcd $objects
    set previous [current_time -s]
    for {set i 0} {$i<26} {incr i} {
        run 10 us;set now [current_time -s]
        puts "HX PROGRESS time=$now";flush stdout
        if {$now eq $previous} {break};set previous $now
    }
    close_vcd;close_sim;close_project
    puts HX_SIMULATOR_RETURNED
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
