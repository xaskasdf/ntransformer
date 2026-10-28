set out [file normalize $::env(NTRPU_V2_OUT)]
if {[file exists $out]} {error "New trace directory required"}
file mkdir $out
open_vcd $out/guard.vcd
set objects [get_objects /$::env(NTRPU_V2_TOP)/*]
if {![llength $objects]} {error "Missing public testbench objects"}
log_vcd $objects
run all
close_vcd
puts V2_SIMULATOR_RETURNED_REVIEW_REQUIRED
quit
