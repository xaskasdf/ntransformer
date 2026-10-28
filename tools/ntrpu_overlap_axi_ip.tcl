# Same-clock AXI4 write boundaries; no width conversion, CDC or memory model.
proc create_transport_axi_slice {output {name ntrpu_mem_regslice}} {
    create_ip -name axi_register_slice -vendor xilinx.com -library ip -version 2.1 -module_name $name
    set ip [get_ips $name]
    set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {WRITE_ONLY} \
        CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {4} \
        CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1}] $ip
    generate_target all $ip
    report_property -file [file join $output ip_properties.rpt] $ip
    set f [open [file join $output ip_config.tsv] w]
    foreach property [lsort [list_property $ip]] {
        if {[string match CONFIG.* $property] || $property in {IPDEF IP_FILE}} {
            puts $f "$property\t[get_property $property $ip]"
        }
    }
    close $f
    return $ip
}
