# Elaborate all native blocks, then optionally synthesize a separate fresh stage.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out project mode
    if {[llength $argv]!=4 || [file exists $out] || $mode ni {elaborate synth}} {error "SOURCE NEW_OUTPUT OWN_PROJECT MODE required"}
    file mkdir $out
    open_project $project
    foreach ip [get_ips] {
        if {[string first "[file dirname [file normalize $project]]/" [file normalize [get_property IP_FILE $ip]]]!=0} {error "IP outside copied project"}
    }
    # read_adapter already contains the identical read_return_cut definition.
    # Read that module once; preserve both frozen source files in the archive.
    foreach name {ntrpu_axi_bridge_guard_v3_reset_export ntrpu_axi_bridge_hbm_client_reset_export ntrpu_axi_bridge_hbm_clocked ntrpu_axi_bridge_hbm_mem_watch ntrpu_axi_bridge_hbm_source ntrpu_axi_bridge_hbm_cdc ntrpu_axi_bridge_endpoint_client ntrpu_axi_bridge_writer ntrpu_axi_bridge_mapper ntrpu_xdma_local_master ntrpu_bridge_async_fifo ntrpu_hbm_startup ntrpu_hbm_read_return ntrpu_hbm_read_frames ntrpu_hbm_read_adapter ntrpu_hbm_native_rw ntrpu_axi_bridge_hbm_native_join} {
        read_verilog -sv $source/$name.sv
    }
    read_verilog $source/xilinx_axi_pcie_ep_app.v
    set_property top ntrpu_axi_bridge_hbm_native_join [current_fileset]
    set f [open $out/primary_clocks.xdc {WRONLY CREAT EXCL}]
    foreach {port name} {sys_clk_p pcie_refclk mgmt_clk_in management_refclk fabric_ref_clk_in fabric_refclk hbm_ref0_clk_in hbm_reference0_clk hbm_ref1_clk_in hbm_reference1_clk} {
        puts $f "create_clock -name $name -period 10.000 \[get_ports $port\]"
    };close $f
    read_xdc $out/primary_clocks.xdc
    update_compile_order -fileset sources_1
    if {$mode eq "elaborate"} {
        synth_design -rtl -top ntrpu_axi_bridge_hbm_native_join -part xcvu47p-fsvh2892-2-e -flatten_hierarchy none
    } else {
        synth_design -top ntrpu_axi_bridge_hbm_native_join -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
        if {[llength [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black box"}
        if {[llength [get_cells -hier -quiet -filter {REF_NAME =~ LD*}]]} {error "Unexpected latch"}
        write_checkpoint $out/synth.dcp
        check_timing -verbose -file $out/check_timing.rpt
        report_cdc -details -file $out/cdc.rpt
        report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
        report_drc -file $out/drc.rpt
        report_utilization -hierarchical -file $out/utilization.rpt
        report_clocks -file $out/clocks.rpt
        write_xdc -constraints all $out/effective.xdc
    }
    set f [open $out/native_instances.tsv {WRONLY CREAT EXCL}];puts $f "cell\treference"
    foreach name {ntrpu_bridge_i hbm_binding/memory clock_generator core} {
        set cell [get_cells -quiet $name]
        if {[llength $cell]!=1} {error "Missing native instance: $name"}
        puts $f "$name\t[get_property REF_NAME $cell]"
    };close $f
    set f [open $out/ports.tsv {WRONLY CREAT EXCL}];puts $f "port\tdirection"
    foreach p [lsort [get_ports]] {puts $f "$p\t[get_property DIRECTION [get_ports $p]]"};close $f
    report_compile_order -used_in synthesis -file $out/compile_order.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_NATIVE_JOIN_${mode}_EXPORTED;close $f
    close_design
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
