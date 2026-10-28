# Private co-generation for structural integration. No example board pinout.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out
    if {[llength $argv]!=2 || [file exists $out]} {error "SOURCE NEW_OUTPUT required"}
    file mkdir $out
    create_project native_join $out/project -part xcvu47p-fsvh2892-2-e
    foreach name {ntrpu_bridge native_hbm axi_bram_ctrl_0} {
        import_ip -files $source/$name.xci
        set ip [get_ips $name]
        set owned [file normalize [get_property IP_FILE $ip]]
        if {[string first "[file normalize $out]/" $owned]!=0} {error "Import escaped private project"}
    }
    set_property CONFIG.pipe_sim false [get_ips ntrpu_bridge]
    foreach {key value} {functional_mode AXI_Bridge axisten_freq 250 axi_addr_width 64 axi_data_width 512_bit pipe_sim false} {
        if {[get_property CONFIG.$key [get_ips ntrpu_bridge]] ne $value} {error "Wrong bridge $key"}
    }
    foreach {key value} {USER_HBM_DENSITY 16GB USER_HBM_STACK 2 USER_AXI_CLK_FREQ 450 USER_AXI_CLK1_FREQ 450 USER_HBM_REF_CLK_0 100 USER_HBM_REF_CLK_1 100 USER_APB_PCLK_0 100 USER_APB_PCLK_1 100} {
        if {[get_property CONFIG.$key [get_ips native_hbm]] ne $value} {error "Wrong memory $key"}
    }
    create_ip -vlnv xilinx.com:ip:clk_wiz:6.0 -module_name fabric_clocks
    set_property -dict [list CONFIG.PRIM_IN_FREQ 100.000 CONFIG.CLKOUT1_REQUESTED_OUT_FREQ 200.000 \
        CONFIG.CLKOUT2_USED true CONFIG.CLKOUT2_REQUESTED_OUT_FREQ 450.000 CONFIG.NUM_OUT_CLKS 2 \
        CONFIG.USE_LOCKED true CONFIG.USE_RESET true CONFIG.RESET_TYPE ACTIVE_HIGH] [get_ips fabric_clocks]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name hbm_read_cdc
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 34 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] [get_ips hbm_read_cdc]
    create_ip -vlnv xilinx.com:ip:axi_dwidth_converter:2.1 -module_name hbm_read_width
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.ADDR_WIDTH 34 CONFIG.SI_DATA_WIDTH 512 CONFIG.MI_DATA_WIDTH 256 CONFIG.SI_ID_WIDTH 4 CONFIG.MAX_SPLIT_BEATS 16 CONFIG.ACLK_ASYNC 0 CONFIG.FIFO_MODE 0] [get_ips hbm_read_width]
    create_ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 -module_name hbm_read_protocol
    set_property -dict [list CONFIG.SI_PROTOCOL AXI4 CONFIG.MI_PROTOCOL AXI3 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.ADDR_WIDTH 34 CONFIG.DATA_WIDTH 256 CONFIG.ID_WIDTH 4 CONFIG.TRANSLATION_MODE 2] [get_ips hbm_read_protocol]
    foreach name {ntrpu_bridge native_hbm axi_bram_ctrl_0 fabric_clocks hbm_read_cdc hbm_read_width hbm_read_protocol} {
        set ip [get_ips $name]
        set f [open $out/${name}_config.tsv {WRONLY CREAT EXCL}]
        foreach key [lsort [list_property $ip]] {
            if {[string match CONFIG.* $key]} {puts $f "$key\t[get_property $key $ip]"}
        };close $f
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        generate_target {instantiation_template synthesis} $ip
    }
    report_ip_status -file $out/ip_status.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_NATIVE_JOIN_GENERATED;close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
