# Logical composition and canonical-reset connectivity only; no timing closure.
set_param general.maxThreads 2
proc hc_one {collection label} {
    if {[llength $collection]!=1} {error "Expected one $label, got [llength $collection]"}
    return [lindex $collection 0]
}
if {[catch {
    lassign $argv source out
    if {[llength $argv]!=2 || [file exists $out]} {error "SOURCE NEW_OUTPUT required"}
    file mkdir $out
    create_project hbm_clocked $out/project -part xcvu47p-fsvh2892-2-e
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name hbm_read_cdc
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 34 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] [get_ips hbm_read_cdc]
    create_ip -vlnv xilinx.com:ip:axi_dwidth_converter:2.1 -module_name hbm_read_width
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.ADDR_WIDTH 34 CONFIG.SI_DATA_WIDTH 512 CONFIG.MI_DATA_WIDTH 256 CONFIG.SI_ID_WIDTH 4 CONFIG.MAX_SPLIT_BEATS 16 CONFIG.ACLK_ASYNC 0 CONFIG.FIFO_MODE 0] [get_ips hbm_read_width]
    create_ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 -module_name hbm_read_protocol
    set_property -dict [list CONFIG.SI_PROTOCOL AXI4 CONFIG.MI_PROTOCOL AXI3 CONFIG.READ_WRITE_MODE READ_ONLY CONFIG.ADDR_WIDTH 34 CONFIG.DATA_WIDTH 256 CONFIG.ID_WIDTH 4 CONFIG.TRANSLATION_MODE 2] [get_ips hbm_read_protocol]
    foreach name {hbm_read_cdc hbm_read_width hbm_read_protocol} {
        set ip [get_ips $name]
        set f [open $out/${name}_config.tsv w]
        foreach key [lsort [list_property $ip]] {
            if {[string match CONFIG.* $key]} {puts $f "$key\t[get_property $key $ip]"}
        };close $f
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        generate_target synthesis $ip
    }
    foreach name {ntrpu_axi_bridge_guard_v3_reset_export ntrpu_axi_bridge_hbm_client_reset_export ntrpu_axi_bridge_hbm_clocked ntrpu_axi_bridge_hbm_mem_watch ntrpu_axi_bridge_hbm_source ntrpu_axi_bridge_hbm_cdc ntrpu_axi_bridge_endpoint_client ntrpu_axi_bridge_writer ntrpu_axi_bridge_mapper ntrpu_xdma_local_master ntrpu_bridge_async_fifo ntrpu_hbm_startup ntrpu_hbm_read_return ntrpu_hbm_read_frames ntrpu_hbm_read_adapter ntrpu_hbm_read_return_cut} {
        read_verilog -sv $source/$name.sv
    }
    set f [open $out/clocks.xdc w]
    foreach {port name period} {clk pcie_clk 4.000 reader_clk reader_clk 5.000 mgmt_clk mgmt_clk 10.000 mem_clk mem_clk 2.222} {
        puts $f "create_clock -name $name -period $period \[get_ports $port\]"
    };close $f
    read_xdc $out/clocks.xdc
    synth_design -top ntrpu_axi_bridge_hbm_clocked -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    if {[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black box"}
    if {[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]} {error "Unexpected latch"}
    write_checkpoint $out/synth.dcp
    set f [open $out/canonical_resets.tsv w]
    puts $f "domain\tconsumer\tdriver"
    foreach {domain cell consumers} [list \
        pcie {endpoint/guard/cold_a_reg[2]} {transport/p_cold_rst_n memory_watch/p_cold_rst_n endpoint/client/cold_rst_n endpoint/source_join/cold_rst_n} \
        management {endpoint/guard/cold_m_reg[2]} {transport/m_cold_rst_n memory_watch/m_cold_rst_n} \
        reader {reader/startup/source_release_reg[2]} {transport/r_cold_rst_n memory_watch/r_cold_rst_n reader/channels[0].frame/rst_n reader/channels[1].frame/rst_n reader/channels[0].crossing/s_axi_aresetn reader/channels[1].crossing/s_axi_aresetn} \
        memory {reader/startup/memory_release_reg[2]} {memory_watch/mem_cold_rst_n reader/channels[0].crossing/m_axi_aresetn reader/channels[1].crossing/m_axi_aresetn reader/channels[0].adapter/rst_n reader/channels[1].adapter/rst_n}] {
        set driver [hc_one [get_pins -quiet $cell/Q] "$domain canonical driver"]
        foreach path $consumers {
            set pin [hc_one [get_pins -quiet $path] $path]
            set nets [get_nets -segments -of_objects $pin]
            set actual [hc_one [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}] "$path leaf driver"]
            if {$actual ne $driver} {error "Noncanonical reset at $path: $actual"}
            puts $f "$domain\t$path\t$actual"
        }
    };close $f
    report_utilization -hierarchical -file $out/utilization.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_cdc -details -file $out/cdc.rpt
    report_drc -file $out/drc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/check_timing.rpt];set checks [read $f];close $f
    foreach kind {loops latch_loops no_clock multiple_clock unconstrained_internal_endpoints} {
        if {![regexp "checking $kind \\((\\d+)\\)" $checks match count] || $count!=0} {error "Unmet structural check: $kind"}
    }
    set f [open $out/STATUS.txt w];puts $f HBM_CLOCKED_CANONICAL_SYNTH_EXPORTED;close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
