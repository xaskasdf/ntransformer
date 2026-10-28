# V4 isolated DMA/PCIe exploration; memory-bank compatibility remains pending.
# Usage: -tclargs EXACT_FPGA_PART NEW_DIRECTORY
proc ntrpu_pcie_probe {arguments} {
    if {[llength $arguments] ni {2 3} || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs EXACT_FPGA_PART NEW_DIRECTORY ?pipe|pipe-msix?"
    }
    lassign $arguments part output mode
    if {$mode ni {{} pipe pipe-msix}} { error "Unknown mode: $mode" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    file mkdir $output
    create_project ntrpu_pcie_probe [file join $output project] -part $part
    create_ip -name xdma -vendor xilinx.com -library ip -version 4.2 -module_name ntrpu_xdma
    set ip [get_ips ntrpu_xdma]
    set requested [list CONFIG.mode_selection Advanced CONFIG.functional_mode DMA \
        CONFIG.pl_link_cap_max_link_width X16 CONFIG.pl_link_cap_max_link_speed 8.0_GT/s \
        CONFIG.pcie_blk_locn X1Y0 CONFIG.select_quad GTY_Quad_227 \
        CONFIG.axi_data_width 512_bit CONFIG.axi_addr_width 64 \
        CONFIG.xdma_axi_intf_mm AXI_Memory_Mapped CONFIG.ref_clk_freq 100_MHz \
        CONFIG.axilite_master_en true CONFIG.axilite_master_scale Kilobytes \
        CONFIG.axilite_master_size 64 CONFIG.enable_tandem_dfx false]
    if {$mode in {pipe pipe-msix}} { lappend requested CONFIG.pipe_sim true }
    if {$mode eq "pipe-msix"} {
        lappend requested CONFIG.pf0_msix_enabled true CONFIG.pf0_msix_cap_table_size 01F \
            CONFIG.pf0_msix_cap_table_bir BAR_1 CONFIG.pf0_msix_cap_table_offset 00008000 \
            CONFIG.pf0_msix_cap_pba_bir BAR_1 CONFIG.pf0_msix_cap_pba_offset 00008FE0
    }
    set_property -dict $requested $ip
    report_property -all -file [file join $output configured_properties.rpt] $ip
    foreach {property expected} $requested {
        if {[get_property $property $ip] ne $expected} {
            error "Requested $property=$expected, got [get_property $property $ip]"
        }
    }
    report_ip_status -file [file join $output ip_status.rpt]
    generate_target all $ip
    help open_example_project
    open_example_project -dir [file join $output example] $ip
    puts NTRPU_PCIE_EXAMPLE_GENERATED
    if {$mode in {pipe pipe-msix}} { puts "NTRPU_PIPE_SIMULATION_ONLY: never implement this variant" }
    puts "No link simulation, synthesis, routing, PERST voltage or DDR coexistence qualification."
    close_project
}
if {[catch {ntrpu_pcie_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
