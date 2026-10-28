set source [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set_param general.maxThreads 2
create_project -in_memory -part xcvu47p-fsvh2892-2-e
read_verilog -sv $source/ntrpu_hbm_weight_loader.sv
synth_design -top ntrpu_hbm_weight_loader -mode out_of_context -flatten_hierarchy none
create_clock -name control -period 4.000 [get_ports clk]
create_clock -name memory -period 2.222 [get_ports mem_clk]
# Deliberately no CDC exceptions: this report exposes unqualified crossings.
report_utilization -file $out/utilization.rpt
report_timing_summary -report_unconstrained -file $out/timing.rpt
report_cdc -details -file $out/cdc.rpt
report_clocks -file $out/clocks.rpt
set f [open $out/primitives.tsv {WRONLY CREAT EXCL}]
foreach cell [lsort [get_cells -hierarchical]] {
    if {[get_property IS_PRIMITIVE $cell]} {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property ASYNC_REG $cell]"
    }
}
close $f
write_checkpoint $out/loader.dcp
puts WEIGHT_LOADER_OOC_SYNTH_COMPLETE
