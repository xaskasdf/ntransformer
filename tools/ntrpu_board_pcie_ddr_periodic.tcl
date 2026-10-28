# Simulation-only PCIe root -> XDMA -> guarded supervised FFN -> generic MIG/DDR.
set script_dir [file dirname [file normalize [info script]]]
proc pcie_ffn {arguments} {
    global script_dir
    if {[llength $arguments] != 5} {error "Expected PCIE_EXAMPLE DDR_EXAMPLE MAC_RUN SWIGLU_RUN NEW_DIRECTORY"}
    lassign $arguments input ddr_example mac activation output
    set mode concurrent;set profile fast;set fixture {}
    if {$mode eq ""} {set mode serialized}
    if {$mode ni {serialized concurrent}} {error "Unknown MMIO/DMA mode"}
    if {$profile eq ""} {set profile typical}
    if {$profile ni {typical fast}} {error "Unknown simulation profile"}
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set bench [file normalize [file join $script_dir .. fpga tb]]
    if {$fixture ne ""} {
        set fixture [file normalize $fixture]
        set bench [file join $output bench]
        exec python3 [file join $script_dir prepare_ntrpu_pcie_ffn_brandon.py] $fixture $bench > [file join $output prepare.log] 2>@1
    }
    file copy $input [file join $output example]
    set patch_command [list python3 [file join $script_dir patch_ntrpu_pcie_ddr_ffn.py] [file join $output example]]
    if {$fixture ne ""} {lappend patch_command --bench $bench}
    exec {*}$patch_command > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    # The copied logger uses a SystemVerilog time literal to report absolute ps
    # independently of compilation-unit timescale. Its parser has no delays.
    set_property file_type SystemVerilog [get_files *usp_pci_exp_usrapp_com.v]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected part"}
    foreach file [get_files -all] {
        if {[string first $input [file normalize $file]]==0} {error "Reference to immutable example: $file"}
    }
    set xdma [get_ips ntrpu_xdma]
    foreach {property expected} {
        CONFIG.pf0_msix_enabled true CONFIG.pf0_msix_cap_table_size 01F
        CONFIG.pf0_msix_cap_table_bir BAR_1 CONFIG.pf0_msix_cap_table_offset 00008000
        CONFIG.pf0_msix_cap_pba_bir BAR_1 CONFIG.pf0_msix_cap_pba_offset 00008FE0
    } {
        if {[get_property $property $xdma] ne $expected} {error "MSI-X input configuration mismatch: $property"}
    }
    report_property -all -file [file join $output xdma_properties.rpt] $xdma
    file copy [file normalize $ddr_example] [file join $output ddr_example]
    file copy [file join $script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
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
    foreach hls [list $mac $activation] kernel {q8_block_mac swiglu_q8_block} {
        set hls [file normalize $hls]
        set f [open [file join $hls manifest.json] r];set text [read $f];close $f
        foreach pattern [list {*"result": "tool_succeeded"*} "*\"kernel\": \"$kernel\"*" {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}] {
            if {![string match $pattern $text]} {error "HLS manifest mismatch"}
        }
        set rtl [file join $output ${kernel}_rtl]
        file copy [file join $hls project solution1 syn verilog] $rtl
        file copy [file join $hls manifest.json] [file join $output ${kernel}_manifest.json]
        cd $rtl
        add_files [glob *.v]
        set data [glob -nocomplain *.dat];if {[llength $data]} {add_files $data}
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {ntrpu_fp64_to_fp32 ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory ntrpu_activation_writer ntrpu_swiglu_memory ntrpu_ffn_sequencer ntrpu_ffn_memory ntrpu_ffn_control ntrpu_irq_transport ntrpu_ffn_irq_control ntrpu_ffn_device ntrpu_axi_arbiter ntrpu_ffn_fabric ntrpu_ffn_async_fabric ntrpu_axil_cdc_recode} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    # Isolated quad/FIFO experiment; original PCIe runner remains unchanged.
    foreach name {ntrpu_q8_gemv_quad ntrpu_q8_gemv_quad_memory ntrpu_reset_supervisor ntrpu_ffn_supervised_fabric ntrpu_axi_capacity_guard ntrpu_ffn_guarded_fabric} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    foreach fileset {sources_1 sim_1} {
        set definitions [get_property verilog_define [get_filesets $fileset]]
        lappend definitions NTRPU_GEMV_QUAD NTRPU_QUAD_BLOCK_FIFO NTRPU_SUPERVISED_FFN
        set_property verilog_define $definitions [get_filesets $fileset]
    }
    puts NTRPU_PCIE_QUAD_FIFO_SELECTED
    foreach {name protocol width address} {ntrpu_host_axi_cdc AXI4 512 64 ntrpu_host_axil_cdc AXI4LITE 32 32} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
        set_property -dict $config $ip;generate_target all $ip
    }
    report_ip_status -file [file join $output ip_status.rpt]
    foreach name {ntrpu_pcie_ddr_periodic_app ntrpu_pcie_ffn_host} {
        add_files -fileset sim_1 [file join $bench ${name}.sv]
    }
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level [expr {$profile eq "fast" ? "off" : "typical"}] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set options {-testplusarg TESTNAME=ntrpu_ffn_msix}
    if {$mode eq "concurrent"} {append options { -testplusarg CONCURRENT_DMA}}
    set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    launch_simulation
    set previous_time [current_time -s]
    set max_steps [expr {$fixture eq "" ? 1001 : 10001}]
    for {set step 0} {$step<$max_steps} {incr step} {
        run 1 us
        set now [current_time -s]
        if {$profile eq "typical"} {
            puts "NTRPU_PCIE_FFN_PROGRESS time=$now ep_link=[get_value -radix bin /board/EP/user_lnk_up] rp_link=[get_value -radix bin /board/RP/user_lnk_up]"
        } else {puts "NTRPU_PCIE_FFN_PROGRESS time=$now profile=fast"}
        flush stdout
        if {$now eq $previous_time} {break}
        set previous_time $now
    }
    close_sim
    set log [file join $output example ntrpu_xdma_ex.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output simulation.log]
    set f [open $log r];set text [read $f];close $f
    set marker [expr {$fixture eq "" ?
        {^NTRPU_PCIE_FFN_PASS cases=3 computed=1 rejected=2 h2c_bytes=3328 core_bytes=704 c2h_bytes=128 msix=2$} :
        {^NTRPU_PCIE_FFN_PASS cases=3 computed=1 rejected=2 h2c_bytes=592384 core_bytes=14016 c2h_bytes=1024 msix=2$}}]
    if {[regexp -all -line $marker $text]!=1 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "PCIe FFN test failed; inspect complete log"}
    if {[regexp -all -line {^NTRPU_PCIE_DDR_PASS } $text]!=1} {error "Missing complete MIG evidence"}
    close_project
    if {[regexp -all -line {^NTRPU_DDR_PERIODIC_PASS } $text]!=1} {error "Missing periodic read witness"}
    puts NTRPU_PCIE_DDR_PERIODIC_SUITE_RETURNED
}
if {[catch {pcie_ffn $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
