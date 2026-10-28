# Simulation-only PCIe root model -> XDMA -> ABI2 FFN -> bounded RAM.
set script_dir [file dirname [file normalize [info script]]]
proc pcie_ffn {arguments} {
    global script_dir
    if {[llength $arguments] ni {4 5 6 7}} {error "Expected EXAMPLE MAC_RUN SWIGLU_RUN NEW_DIRECTORY ?serialized|concurrent? ?typical|fast? ?BRANDON_FIXTURE?"}
    lassign $arguments input mac activation output mode profile fixture
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
    set patch_command [list python3 [file join $script_dir patch_ntrpu_pcie_ffn.py] [file join $output example]]
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
    foreach {name protocol width address} {ntrpu_host_axi_cdc AXI4 512 64 ntrpu_host_axil_cdc AXI4LITE 32 32} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
        set_property -dict $config $ip;generate_target all $ip
    }
    report_ip_status -file [file join $output ip_status.rpt]
    foreach name {ntrpu_pcie_ffn_app ntrpu_pcie_ffn_host} {
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
    close_project
    puts NTRPU_PCIE_FFN_SUITE_RETURNED
}
if {[catch {pcie_ffn $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
