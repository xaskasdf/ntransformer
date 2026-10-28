# Standalone pad/sequencer probe; not synthesis of the native PCIe/HBM composition.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out
    file mkdir $out
    read_verilog -sv [list $source/ntrpu_hbm_board_inputs.sv $source/ntrpu_hbm_board_startup.sv]
    set f [open $out/frontend_probe.sv w]
    puts $f {
module frontend_probe(
 input wire mgmt_clk_p,mgmt_clk_n,hbm_ref0_clk_p,hbm_ref0_clk_n,hbm_ref1_clk_p,hbm_ref1_clk_n,
 input wire supervisor_rst_n_pad,fabric_locked_async,
 output wire mgmt_clk,hbm_ref0,hbm_ref1,fabric_clock_reset,configuration_rst_n,lock_fault);
 wire mgmt_clk_in,hbm_ref0_clk_in,hbm_ref1_clk_in,supervisor_rst_n;
 ntrpu_hbm_board_inputs board_inputs(.*);
 BUFGCE management_buffer(.I(mgmt_clk_in),.CE(1'b1),.O(mgmt_clk));
 BUFGCE hbm_reference0(.I(hbm_ref0_clk_in),.CE(1'b1),.O(hbm_ref0));
 BUFGCE hbm_reference1(.I(hbm_ref1_clk_in),.CE(1'b1),.O(hbm_ref1));
 ntrpu_hbm_board_startup startup(.*);
endmodule
    };close $f
    read_verilog -sv $out/frontend_probe.sv
    synth_design -top frontend_probe -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy none
    create_clock -name mgmt_ref -period 10 [get_ports mgmt_clk_p]
    create_clock -name hbm_ref0 -period 10 [get_ports hbm_ref0_clk_p]
    create_clock -name hbm_ref1 -period 10 [get_ports hbm_ref1_clk_p]
    if {[llength [get_cells -hier -filter {IS_BLACKBOX == 1}]]} {error "Black boxes"}
    if {[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]} {error "Latches"}
    write_checkpoint $out/frontend.dcp
    report_utilization -hierarchical -file $out/utilization.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_drc -file $out/drc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    set f [open $out/primitives.tsv w];puts $f "cell\tref"
    foreach c [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]] {puts $f "$c\t[get_property REF_NAME $c]"};close $f
    set f [open $out/STATUS.txt w];puts $f BOARD_FRONTEND_STANDALONE_SYNTHESIZED;close $f
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
