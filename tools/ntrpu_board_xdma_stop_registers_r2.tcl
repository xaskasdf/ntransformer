set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 2
proc simulate_stop_registers {root output} {
    if {[file exists $output]} {error "Output exists"}
    create_project stop_registers [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [list [file join $root fpga rtl ntrpu_xdma_stop_registers_r1.sv] [file join $root fpga rtl ntrpu_xdma_local_master.sv]]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_xdma_stop_registers_r2.sv]
    set_property top tb_ntrpu_xdma_stop_registers [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set names {clk cold_rst_n core_ready exclusive_owner external_fault wait_expired start_valid start_ready
        start_generation start_tag close_request host_closed host_generation host_tag ledger_active ledger_closed
        ledger_accounted ledger_generation ledger_tag ledger_published ledger_completed ledger_faults release_valid
        active generation tag expected_published faults snapshot_valid control_snapshot status_snapshot count_snapshot
        stop_responses read_responses transaction_pending m_admit m_req_valid m_req_ready m_req_write m_req_address
        m_req_data m_req_strobe m_req_identity m_rsp_valid m_rsp_ready m_rsp_data m_rsp_code m_rsp_identity
        m_rsp_timeout m_timeout_latched allow_request wrong_identity unsolicited master_req_ready master_rsp_valid
        master_rsp_identity master_busy awaddr wdata araddr wstrb awprot arprot awvalid wvalid bready arvalid rready
        awready wready bvalid arready rvalid bresp rresp rdata hold_b hold_r h_bcode c_bcode read_code
        h_control c_control h_status c_status h_count c_count phase cycle write_count read_count}
    set objects {}
    foreach name $names {
        set found [get_objects /tb_ntrpu_xdma_stop_registers/$name]
        if {[llength $found]!=1} {error "Missing public test signal $name"}
        lappend objects {*}$found
    }
    open_vcd [file join $output stop_registers.vcd]
    log_vcd $objects
    run all
    close_vcd
    close_sim
    set logpath [file join $output project stop_registers.sim sim_1 behav xsim simulate.log]
    file copy $logpath [file join $output simulate.log]
    set f [open $logpath r];set log [read $f];close $f
    if {[lsearch -exact [split $log \n] "STOP_REG_PASS cases=89 master_wait_limit=32 native_xdma=0 global_drain=0 reset_permission=0"]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $log]} {error "Stop register tests failed"}
    close_project
    puts "NTRPU_XDMA_STOP_REGISTERS_RETURNED"
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {simulate_stop_registers $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
