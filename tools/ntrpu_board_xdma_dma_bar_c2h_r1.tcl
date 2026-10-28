# Native local management under retained BAR B, H2C B and C2H R. No reset.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
set_param general.maxThreads 2
proc ntrpu_stop_inventory {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {error "Use fresh process: SEED_EXAMPLE NEW_OUTPUT"}
    lassign $arguments input output
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output] || ![file exists [file join $input ntrpu_xdma_ex.xpr]]} {error "Invalid paths"}
    file mkdir $output
    file copy $input [file join $output example]
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_xdma_local_management_r1.py] [file join $output example] > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first $input [file normalize $f]]==0} {error "Reference to immutable seed: $f"}
    }
    foreach {key value} {CONFIG.functional_mode DMA CONFIG.soft_reset_en true CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axilite_master_en true CONFIG.xdma_dsc_bypass false CONFIG.xdma_sts_ports true CONFIG.xdma_axilite_slave true CONFIG.xdma_rnum_chnl 1 CONFIG.xdma_wnum_chnl 1 CONFIG.comp_timeout 50ms} {
        if {[get_property $key [get_ips ntrpu_xdma]] ne $value} {error "Wrong $key"}
    }
    report_property -all -file [file join $output xdma_properties.rpt] [get_ips ntrpu_xdma]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_xdma_dma_bar_c2h.sv]
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_xdma_local_master.sv]
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_xdma_descriptor_ledger.sv]
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_xdma_stop_registers_r1.sv]
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_drain_monitor_r2.sv]
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options {-testplusarg TESTNAME=ntrpu_stop_inventory}] [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    report_ip_status -file [file join $output ip_status.rpt]
    update_compile_order -fileset sim_1
    launch_simulation
    set names {/board/sys_rst_n /board/EP/user_clk /board/EP/user_resetn /board/EP/user_lnk_up /board/RP/user_clk /board/RP/user_lnk_up /board/ntrpu_stop_probe/phase /board/ntrpu_stop_probe/hold_dma_r /board/ntrpu_stop_probe/host_memory /board/ntrpu_stop_probe/host_written /board/ntrpu_stop_probe/host_bytes /board/ntrpu_stop_probe/host_packets /board/ntrpu_stop_probe/host_active /board/EP/ntrpu_app_dma_rvalid /board/EP/ntrpu_app_dma_rready /board/ntrpu_stop_probe/hold_dma_b /board/ntrpu_stop_probe/hold_bypass_b /board/EP/ntrpu_app_dma_bvalid /board/EP/ntrpu_app_dma_bready /board/EP/ntrpu_app_bypass_bvalid /board/EP/ntrpu_app_bypass_bready /board/RP/tx_usrapp/P_READ_DATA}
    foreach prefix {m_axi m_axib} {
        foreach suffix {awvalid awready awid awaddr awlen awsize awburst awprot awcache awlock wvalid wready wdata wstrb wlast bvalid bready bid bresp arvalid arready arid araddr arlen arsize arprot arcache arlock arburst rvalid rready rid rdata rresp rlast} {lappend names /board/EP/${prefix}_${suffix}}
    }
    foreach prefix {s_axis_rq m_axis_rc s_axis_cc m_axis_cq} {
        foreach suffix {tvalid tready tdata tkeep tlast tuser} {lappend names /board/RP/${prefix}_${suffix}}
    }
    foreach suffix {awaddr awprot awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready} {
        lappend names /board/ntrpu_stop_probe/local_${suffix}
    }
    foreach signal {dma_aw dma_w dma_b dma_ar dma_r h2c_status c2h_status local_reads local_writes h2c_done_events c2h_done_events init_ar_count init_r_count init_rvalid init_arready} {
        lappend names /board/ntrpu_stop_probe/$signal
    }
    foreach suffix {awvalid araddr arvalid arready rvalid rready rdata rresp} {lappend names /board/EP/m_axil_${suffix}}
    foreach signal {master_admit master_req_valid master_req_ready master_req_write master_req_address master_req_data master_req_strobe master_req_identity master_rsp_valid master_rsp_ready master_rsp_data master_rsp_code master_rsp_identity master_rsp_timeout master_timeout_latched master_busy master_requests master_reports master_serial} {
        lappend names /board/ntrpu_stop_probe/$signal
    }
    foreach signal {ledger_arm_valid ledger_arm_ready ledger_publish_valid ledger_publish_channel
        ledger_publish_index ledger_publish_address ledger_publish_image ledger_h2c_image ledger_c2h_image
        ledger_close_request ledger_host_closed ledger_host_generation ledger_host_tag
        ledger_active ledger_closed ledger_generation ledger_tag ledger_started ledger_faults
        ledger_published ledger_completed ledger_outstanding ledger_accounted ledger_ready} {
        lappend names /board/ntrpu_stop_probe/$signal
    }
    foreach signal {sequencer_owner seq_start_valid seq_start_ready seq_active seq_snapshot_valid seq_pending
        seq_generation seq_tag seq_expected seq_control seq_status seq_count seq_faults seq_stops seq_reads
        seq_admit seq_req_valid seq_req_write seq_rsp_ready seq_req_address seq_req_data seq_req_strobe
        seq_req_identity seq_reports seq_requests} {
        lappend names /board/ntrpu_stop_probe/$signal
    }
    foreach signal {inventory_source_fenced inventory_request} {lappend names /board/ntrpu_stop_probe/$signal}
    foreach prefix {dma bar} {
        foreach suffix {ack tag fault fault_sources aw_pending w_pending ar_pending w_partial} {
            lappend names /board/ntrpu_stop_probe/${prefix}_obs_${suffix}
        }
    }
    lappend names /board/EP/ntrpu_xdma_i/m_axi_rlast /board/EP/ntrpu_xdma_i/m_axib_rlast
    lappend names /board/EP/ntrpu_xdma_i/dma_bridge_resetn
    set objects {}
    set f [open [file join $output trace_objects.txt] {WRONLY CREAT EXCL}]
    foreach name $names {
        set found [get_objects $name]
        if {[llength $found]!=1} {error "Missing native signal: $name"}
        puts $f $name; lappend objects {*}$found
    }
    close $f
    open_vcd [file join $output native.vcd]
    log_vcd $objects
    set previous [current_time -s]
    for {set i 0} {$i<241} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_STOP_INVENTORY_PROGRESS time=$now"
        flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    close_vcd
    close_sim
    close_project
    puts NTRPU_STOP_INVENTORY_SIMULATOR_RETURNED
}
if {[catch {ntrpu_stop_inventory $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
