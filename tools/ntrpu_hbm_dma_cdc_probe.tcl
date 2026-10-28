# Isolated OOC synthesis/CDC inspection. This is not a board timing signoff.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
file mkdir $out
set_param general.maxThreads 2
create_project -in_memory -part xcvu47p-fsvh2892-2-e
foreach name {ntrpu_peer_read_pkg ntrpu_bridge_async_fifo ntrpu_hbm_dma_bridge_lane ntrpu_hbm_dma_bridge} {
    read_verilog -sv [file join $root fpga experimental ${name}.sv]
}
synth_design -top ntrpu_hbm_dma_bridge -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
create_clock -name reader_clk -period 5.000 [get_ports clk]
create_clock -name dma_clk -period 4.000 [get_ports dma_clk]
set_clock_uncertainty -setup 0.100 [get_clocks]
set_clock_uncertainty -hold 0.050 [get_clocks]
report_clocks -file $out/clocks.rpt
report_utilization -hierarchical -file $out/utilization.rpt
report_cdc -details -all_checks_per_endpoint -file $out/cdc_raw.rpt
report_timing_summary -report_unconstrained -file $out/timing_raw.rpt
report_drc -file $out/drc_raw.rpt
write_checkpoint -force $out/synth.dcp
write_verilog -force -mode funcsim $out/synth.v
set f [open $out/cells.tsv w]
puts $f "name\tref\tasync_reg"
foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {
    puts $f "[get_property NAME $c]\t[get_property REF_NAME $c]\t[get_property ASYNC_REG $c]"
}
close $f
set f [open $out/ports.tsv w]
puts $f "name\tdirection"
foreach p [get_ports] {puts $f "[get_property NAME $p]\t[get_property DIRECTION $p]"}
close $f
puts "NTRPU_BRIDGE_SYNTH_INSPECT_DONE"
