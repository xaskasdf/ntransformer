# Isolated VU47P XDMA stream candidate. No existing board configuration is changed.
set output [file normalize [lindex $argv 0]]
set part xcvu47p-fsvh2892-2-e
set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=1 || [file exists $output]} {error "Expected one fresh output directory"}
    file mkdir $output
    foreach profile {serial pipe} {
        set folder [file join $output $profile]
        create_project hbm_pcie_probe [file join $folder project] -part $part
        create_ip -name xdma -vendor xilinx.com -library ip -version 4.2 -module_name ntrpu_vu47p_xdma
        set ip [get_ips ntrpu_vu47p_xdma]
        # Keep Vivado's device-specific block/quad selection; record the resolved
        # locations. This is a legal-IP probe, not an approved connector pinout.
        set requested [list CONFIG.mode_selection Advanced CONFIG.functional_mode DMA \
            CONFIG.pl_link_cap_max_link_width X16 CONFIG.pl_link_cap_max_link_speed 8.0_GT/s \
            CONFIG.axi_data_width 512_bit CONFIG.axi_addr_width 64 \
            CONFIG.xdma_axi_intf_mm AXI_Stream CONFIG.ref_clk_freq 100_MHz \
            CONFIG.xdma_rnum_chnl 1 CONFIG.xdma_wnum_chnl 2 \
            CONFIG.xdma_dsc_bypass true CONFIG.dsc_bypass_rd 0001 CONFIG.dsc_bypass_wr 0011 \
            CONFIG.xdma_sts_ports true CONFIG.xdma_axilite_slave true CONFIG.soft_reset_en false \
            CONFIG.axilite_master_en true CONFIG.axilite_master_scale Kilobytes CONFIG.axilite_master_size 64 \
            CONFIG.enable_tandem_dfx false CONFIG.pipe_sim [expr {$profile eq "pipe" ? "true" : "false"}] \
            CONFIG.pf0_msix_enabled true CONFIG.pf0_msix_cap_table_size 01F \
            CONFIG.pf0_msix_cap_table_bir BAR_1 CONFIG.pf0_msix_cap_table_offset 00008000 \
            CONFIG.pf0_msix_cap_pba_bir BAR_1 CONFIG.pf0_msix_cap_pba_offset 00008FE0 \
            CONFIG.axil_master_64bit_en false CONFIG.xdma_pcie_64bit_en false \
            CONFIG.axist_bypass_en true CONFIG.axist_bypass_size 1 CONFIG.axist_bypass_scale Megabytes \
            CONFIG.axi_bypass_64bit_en true CONFIG.axi_bypass_prefetchable false \
            CONFIG.pciebar2axibar_axist_bypass 0x0000000100000000]
        foreach {key value} $requested {
            if {$key ni [list_property $ip]} {error "Unknown parameter $key"}
        }
        set_property -dict $requested $ip
        set f [open [file join $folder requested.tsv] w];puts $f "property\trequested\tactual"
        foreach {key value} $requested {
            set actual [get_property $key $ip];puts $f "$key\t$value\t$actual"
            if {$actual ne $value} {error "Unhonored $key: requested $value actual $actual"}
        }
        close $f
        set f [open [file join $folder config.tsv] w]
        puts $f "PART\t[get_property PART [current_project]]"
        foreach key [lsort [list_property $ip]] {
            if {[string match CONFIG.* $key]} {puts $f "$key\t[get_property $key $ip]"}
        }
        close $f
        report_property -all -file [file join $folder properties.rpt] $ip
        generate_target all $ip
        report_ip_status -file [file join $folder ip_status.rpt]
        open_example_project -dir [file join $folder example] $ip
        close_project
        puts "HP GENERATED $profile $part"
    }
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts HP_PROBE_COMPLETE
exit 0
