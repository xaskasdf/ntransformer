# Structural synthesis of the reader-facing client, without native HBM/PCIe IP.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out
    if {[llength $argv]!=2 || [file exists $out]} {error "SOURCE NEW_OUTPUT required"}
    file mkdir $out
    create_project hbm_source_client $out/project -part xcvu47p-fsvh2892-2-e
    foreach name {ntrpu_axi_bridge_hbm_source ntrpu_axi_bridge_hbm_client ntrpu_axi_bridge_guard_v3 ntrpu_axi_bridge_endpoint_client ntrpu_axi_bridge_mapper ntrpu_axi_bridge_writer ntrpu_xdma_local_master} {
        read_verilog -sv $source/$name.sv
    }
    set f [open $out/clocks.xdc {WRONLY CREAT EXCL}]
    puts $f {create_clock -name bridge_clk -period 4.000 [get_ports clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    read_xdc $out/clocks.xdc
    synth_design -top ntrpu_axi_bridge_hbm_client -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
    if {[llength [get_cells -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black box"}
    if {[llength [get_cells -hierarchical -filter {REF_NAME =~ LD*}]]} {error "Unexpected latch"}
    write_checkpoint $out/synth.dcp
    report_utilization -hierarchical -file $out/utilization.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_drc -file $out/drc.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/check_timing.rpt r];set timing_checks [read $f];close $f
    foreach kind {loops latch_loops no_clock unconstrained_internal_endpoints} {
        if {![regexp "checking $kind \\((\\d+)\\)" $timing_checks match count] || $count!=0} {error "Unmet structural check: $kind"}
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_SOURCE_CLIENT_SYNTHESIZED_NOT_PHYSICALLY_QUALIFIED
    close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
