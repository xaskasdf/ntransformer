# Independent level/status flags; only the first synchronizer D pins.
# Load after clocks and the synthesized ntrpu_hbm_transport_async hierarchy.
# Never exclude memory_release/source_release -> functional datapath paths.
# Topology/count assertions belong in the Tcl driver; XDC disallows if/foreach.
set ntrpu_status_internal_d [get_pins -quiet -hier -regexp {(^|.*/)startup/(fault_meta_reg\[[0-1]\]|ready_meta_reg\[2\])/D}]
set ntrpu_status_apb_d [get_pins -quiet -hier -regexp {(^|.*/)startup/ready_meta_reg\[[0-1]\]/D}]
set ntrpu_status_budget [get_property PERIOD [get_clocks transport_clk]]
set_max_delay -datapath_only $ntrpu_status_budget -from [get_clocks memory_clk] -to $ntrpu_status_internal_d
set_max_delay -datapath_only $ntrpu_status_budget -from [get_ports init_complete_async*] -to $ntrpu_status_apb_d
