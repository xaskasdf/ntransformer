# V4 isolated BAR-to-AXI bypass generation; PIPE simulation only.
# Does not qualify PCIe transactions, P2P, memory, reset, or board implementation.
# Usage: -tclargs EXACT_FPGA_PART NEW_DIRECTORY
proc ntrpu_pcie_bypass_probe {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs EXACT_FPGA_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    file mkdir $output
    create_project ntrpu_pcie_bypass_probe [file join $output project] -part $part
    create_ip -name xdma -vendor xilinx.com -library ip -version 4.2 -module_name ntrpu_xdma
    set ip [get_ips ntrpu_xdma]
    # Explicitly keep the two existing BARs 32-bit. The new 64-bit bypass
    # aperture is a window, not a declaration of coherent aggregate memory.
    set requested [list CONFIG.mode_selection Advanced CONFIG.functional_mode DMA \
        CONFIG.pl_link_cap_max_link_width X16 CONFIG.pl_link_cap_max_link_speed 8.0_GT/s \
        CONFIG.pcie_blk_locn X1Y0 CONFIG.select_quad GTY_Quad_227 \
        CONFIG.axi_data_width 512_bit CONFIG.axi_addr_width 64 \
        CONFIG.xdma_axi_intf_mm AXI_Memory_Mapped CONFIG.ref_clk_freq 100_MHz \
        CONFIG.axilite_master_en true CONFIG.axilite_master_scale Kilobytes \
        CONFIG.axilite_master_size 64 CONFIG.enable_tandem_dfx false \
        CONFIG.pipe_sim true CONFIG.pf0_msix_enabled true \
        CONFIG.pf0_msix_cap_table_size 01F CONFIG.pf0_msix_cap_table_bir BAR_1 \
        CONFIG.pf0_msix_cap_table_offset 00008000 CONFIG.pf0_msix_cap_pba_bir BAR_1 \
        CONFIG.pf0_msix_cap_pba_offset 00008FE0 \
        CONFIG.axil_master_64bit_en false CONFIG.xdma_pcie_64bit_en false \
        CONFIG.axist_bypass_en true CONFIG.axist_bypass_size 1 \
        CONFIG.axist_bypass_scale Megabytes CONFIG.axi_bypass_64bit_en true \
        CONFIG.axi_bypass_prefetchable false \
        CONFIG.pciebar2axibar_axist_bypass 0x0000000100000000]
    foreach {property expected} $requested {
        if {$property ni [list_property $ip]} { error "Unavailable property: $property" }
    }
    set_property -dict $requested $ip
    report_property -all -file [file join $output configured_properties.rpt] $ip
    set f [open [file join $output requested_properties.tsv] w]
    puts $f "property\trequested\tactual"
    foreach {property expected} $requested {
        set actual [get_property $property $ip]
        puts $f "$property\t$expected\t$actual"
        if {$actual ne $expected} { close $f; error "$property: requested $expected, got $actual" }
    }
    close $f
    report_ip_status -file [file join $output ip_status.rpt]
    generate_target all $ip
    open_example_project -dir [file join $output example] $ip
    puts NTRPU_PCIE_BYPASS_EXAMPLE_GENERATED
    puts "NTRPU_PIPE_SIMULATION_ONLY: never implement this variant"
    puts "Inspect generated hard-block BAR parameters, translation and m_axib ports before traffic simulation."
    puts "No link simulation, P2P, synthesis, routing, PERST voltage or DDR coexistence qualification."
    close_project
}
if {[catch {ntrpu_pcie_bypass_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
