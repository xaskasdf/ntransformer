set source_dir [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
set_param general.maxThreads 2
read_verilog -sv [file join $source_dir ntrpu_control_snapshot.sv]
read_verilog -sv [file join $source_dir ntrpu_hbm_control_plane.sv]
synth_design -top ntrpu_hbm_control_plane -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
create_clock -period 4.000 -name control_clk [get_ports clk]
foreach index {0 1 2 3} period {4.000 10.000 5.000 2.222} {
    create_clock -period $period -name telemetry_$index [get_ports [format {telemetry_clk[%d]} $index]]
}
report_utilization -file [file join $output_dir utilization.rpt]
report_cdc -details -file [file join $output_dir cdc_unconstrained.rpt]
report_timing_summary -file [file join $output_dir timing_unconstrained.rpt]
write_checkpoint [file join $output_dir control.dcp]
set f [open [file join $output_dir structure.tsv] w]
puts $f "kind\tcount"
puts $f "ASYNC_REG\t[llength [get_cells -hier -filter {ASYNC_REG == TRUE}]]"
puts $f "FDRE\t[llength [get_cells -hier -filter {REF_NAME == FDRE}]]"
puts $f "FDCE\t[llength [get_cells -hier -filter {REF_NAME == FDCE}]]"
puts $f "LUT\t[llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
puts $f "LATCH\t[llength [get_cells -hier -filter {REF_NAME =~ LD*}]]"
close $f
puts CONTROL_OOC_SYNTH_COMPLETE
