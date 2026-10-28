# Generate only a fresh VU47P bridge candidate. No simulation or implementation.
# Arguments: a new output directory. Existing projects are never opened/modified.
set out [file normalize [lindex $argv 0]]
set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=1 || [file exists $out]} {error "Expected one new output directory"}
    file mkdir $out
    create_project bridge_probe $out/project -part xcvu47p-fsvh2892-2-e
    create_ip -name xdma -vendor xilinx.com -library ip -version 4.2 -module_name ntrpu_bridge
    set ip [get_ips ntrpu_bridge]
    set requested [list CONFIG.mode_selection Advanced CONFIG.functional_mode AXI_Bridge \
        CONFIG.pl_link_cap_max_link_width X16 CONFIG.pl_link_cap_max_link_speed 8.0_GT/s \
        CONFIG.axi_data_width 512_bit CONFIG.axi_addr_width 64 CONFIG.s_axi_id_width 4 \
        CONFIG.axisten_freq 250 CONFIG.ref_clk_freq 100_MHz CONFIG.parity_settings None \
        CONFIG.xdma_axilite_slave true CONFIG.pipe_sim true CONFIG.axibar_num 1 \
        CONFIG.axibar_0 0x0000000010000000 CONFIG.axibar_highaddr_0 0x0000000010000fff \
        CONFIG.axibar2pciebar_0 0x1234567800001000]
    foreach {key value} $requested {
        if {$key ni [list_property $ip]} {error "Unknown property $key"}
    }
    set_property -dict $requested $ip
    set f [open $out/requested.tsv w]
    puts $f "property\trequested\tactual"
    foreach {key value} $requested {
        set actual [get_property $key $ip]
        puts $f "$key\t$value\t$actual"
        if {[string match 0x* $value]} {
            if {[expr {$value != $actual}]} {error "Unhonored $key"}
        } elseif {$value ne $actual} {error "Unhonored $key: $actual"}
    }
    close $f
    set f [open $out/config.tsv w]
    foreach key [lsort [list_property $ip]] {
        if {[string match CONFIG.* $key]} {puts $f "$key\t[get_property $key $ip]"}
    }
    close $f
    report_property -all -file $out/properties.rpt $ip
    generate_target {synthesis simulation} $ip
    report_ip_status -file $out/ip_status.rpt
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts BRIDGE_NATIVE_PRODUCTS_GENERATED
exit 0
