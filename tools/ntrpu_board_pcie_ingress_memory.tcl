# Native posted PCIe writes -> actual CDC/ingress/capacity/MIG and DDR readback.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_bypass_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=3 || [llength [get_projects -quiet]]} {
        error "Use fresh process: PCIE_EXAMPLE DDR_EXAMPLE NEW_OUTPUT"
    }
    lassign $arguments input ddr_example output
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output] || ![file exists [file join $input ntrpu_xdma_ex.xpr]]} {
        error "Missing input or output already exists"
    }
    file mkdir $output
    file copy $input [file join $output example]
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_ingress_memory.py] [file join $output example] > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first $input [file normalize $f]]==0} {error "Copied project references immutable input: $f"}
    }
    set ip [get_ips ntrpu_xdma]
    foreach {key value} {CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axi_bypass_64bit_en true CONFIG.pciebar2axibar_axist_bypass 0x0000000100000000} {
        if {[get_property $key $ip] ne $value} {error "Wrong $key"}
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_ingress_memory_harness.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_dword_decode.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_line_ingress.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_peer_rx_guard.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga rtl ntrpu_pcie_ingress_peer.sv]
    file copy [file normalize $ddr_example] [file join $output ddr_example]
    file copy [file join $ntrpu_script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    read_ip [file join $output ddr_example ntrpu_ddr_geometry_ex.srcs sources_1 ip ntrpu_ddr_geometry ntrpu_ddr_geometry.xci]
    set ip [get_ips ntrpu_ddr_geometry]
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000
        CONFIG.C0.DDR4_CasLatency 12 CONFIG.C0.DDR4_CasWriteLatency 9
    } {
        if {[get_property $property $ip] ne $expected} {
            error "Unexpected $property: [get_property $property $ip]; expected $expected"
        }
    }
    report_property -all -file [file join $output controller_properties.rpt] $ip
    generate_target all $ip

    # These definitions follow AMD ddr4_v2_2/hdl/model_wrapper.ttcl, including
    # its DDR4_1250_Timing -> FIXED_1600 branch. The model itself stays intact.
    set model_path [file join $output ddr_example imports ddr4_sdram_model_wrapper.sv]
    set stream [open $model_path r]; set model [read $stream]; close $stream
    foreach {before after} {
        DDR4_4G_X8 DDR4_16G_X4
        DDR4_750_Timing DDR4_1250_Timing
        FIXED_2666 FIXED_1600
    } {
        if {[regexp -all $before $model] != 1} { error "Unexpected model template: $before" }
        set model [string map [list $before $after] $model]
    }
    set stream [open $model_path w]; puts -nonewline $stream $model; close $stream
    set model_dir [file join $output ddr_example imports]
    foreach name {MemoryArray.sv arch_defines.v arch_package.sv ddr4_bi_delay.sv ddr4_db_delay_model.sv ddr4_dimm.sv ddr4_dir_detect.sv ddr4_sdram_model_wrapper.sv proj_package.sv timing_tasks.sv StateTableCore.sv StateTable.sv ddr4_model.sv ddr4_rank.sv ddr4_rcd_model.sv ddr4_rdimm_wrapper.sv interface.sv temp_mem.txt temp_second_mem.txt} {
        add_files -fileset sim_1 [file join $model_dir $name]
    }
    foreach name {MemoryArray.sv arch_defines.v timing_tasks.sv StateTableCore.sv StateTable.sv} {
        set_property file_type {Verilog Header} [get_files [file join $model_dir $name]]
    }
    set_property include_dirs [concat [get_property include_dirs [get_filesets sim_1]] [list $model_dir]] [get_filesets sim_1]
    set_property XPM_LIBRARIES XPM_CDC [current_project]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_ingress_axi_cdc
    set cdc [get_ips ntrpu_ingress_axi_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $cdc]]
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $cdc
    generate_target all $cdc
    report_property -all -file [file join $output cdc_properties.rpt] $cdc
    report_ip_status -file [file join $output ip_status.rpt]
    foreach name {ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
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
foreach suffix {clk rst_n host_rst_n calibrated ui_rst guard_idle} {lappend names /board/ntrpu_bypass/$suffix}
foreach suffix {awaddr awid awlen awsize awburst awlock awvalid awready wdata wstrb wlast wvalid wready bid bresp bvalid bready} {
    lappend names /board/ntrpu_bypass/dut/d_axi_$suffix
}
foreach suffix {araddr arid arlen arsize arburst arlock arcache arprot arvalid arready rdata rid rresp rlast rvalid rready} {
    lappend names /board/ntrpu_bypass/m_axi_$suffix
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
    for {set i 0} {$i<305} {incr i} {
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
    puts NTRPU_NATIVE_MIG_SIMULATOR_RETURNED
}
if {[catch {ntrpu_bypass_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
