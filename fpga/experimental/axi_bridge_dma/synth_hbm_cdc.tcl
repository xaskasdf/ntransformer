# Structure and unwaived CDC inventory only. No physical qualification.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out
    if {[llength $argv]!=2 || [file exists $out]} {error "SOURCE NEW_OUTPUT required"}
    file mkdir $out
    create_project hbm_cdc $out/project -part xcvu47p-fsvh2892-2-e
    read_verilog -sv $source/ntrpu_bridge_async_fifo.sv
    read_verilog -sv $source/ntrpu_axi_bridge_hbm_cdc.sv
    set f [open $out/clocks.xdc {WRONLY CREAT EXCL}]
    puts $f {create_clock -name pcie_clk -period 4.000 [get_ports p_clk]}
    puts $f {create_clock -name reader_clk -period 5.000 [get_ports r_clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    read_xdc $out/clocks.xdc
    synth_design -top ntrpu_axi_bridge_hbm_cdc -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    if {[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black box"}
    if {[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]} {error "Unexpected latch"}
    write_checkpoint $out/synth.dcp
    report_utilization -hierarchical -file $out/utilization.rpt
    check_timing -verbose -file $out/check_timing.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_cdc -details -file $out/cdc.rpt
    report_drc -file $out/drc.rpt
    write_xdc -constraints all $out/effective.xdc
    set f [open $out/check_timing.rpt r];set checks [read $f];close $f
    foreach kind {loops latch_loops no_clock multiple_clock unconstrained_internal_endpoints} {
        if {![regexp "checking $kind \\((\\d+)\\)" $checks match count] || $count!=0} {error "Unmet structural check: $kind"}
    }
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CDC_SYNTHESIZED_NOT_PHYSICALLY_QUALIFIED
    close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
