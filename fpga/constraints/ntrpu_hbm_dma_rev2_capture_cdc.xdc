# Revision 2 controls plus bounded RAM-to-output-register capture paths.
# Validate exact bindings with tools/ntrpu_hbm_dma_rev2_capture_review.tcl.
# Not a complete FIFO/data/reset/interface CDC qualification. No waivers.
# reader_clk=5ns, dma_clk=4ns; this exact profile is checked by the driver.

set bridge_l0_commands_wg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/commands/wg_reg\[[0-2]\]}]
set bridge_l0_commands_wg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/commands/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l0_commands_wg_from -to $bridge_l0_commands_wg_to
set_bus_skew 4.000 -from $bridge_l0_commands_wg_from -to $bridge_l0_commands_wg_to

set bridge_l0_commands_rg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/commands/rg_reg\[[0-2]\]}]
set bridge_l0_commands_rg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/commands/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_commands_rg_from -to $bridge_l0_commands_rg_to
set_bus_skew 4.000 -from $bridge_l0_commands_rg_from -to $bridge_l0_commands_rg_to

set bridge_l0_payloads_wg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/payloads/wg_reg\[[0-2]\]}]
set bridge_l0_payloads_wg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/payloads/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l0_payloads_wg_from -to $bridge_l0_payloads_wg_to
set_bus_skew 4.000 -from $bridge_l0_payloads_wg_from -to $bridge_l0_payloads_wg_to

set bridge_l0_payloads_rg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/payloads/rg_reg\[[0-2]\]}]
set bridge_l0_payloads_rg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/payloads/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_payloads_rg_from -to $bridge_l0_payloads_rg_to
set_bus_skew 4.000 -from $bridge_l0_payloads_rg_from -to $bridge_l0_payloads_rg_to

set bridge_l0_completions_wg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/completions/wg_reg\[[0-2]\]}]
set bridge_l0_completions_wg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/completions/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_completions_wg_from -to $bridge_l0_completions_wg_to
set_bus_skew 4.000 -from $bridge_l0_completions_wg_from -to $bridge_l0_completions_wg_to

set bridge_l0_completions_rg_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/completions/rg_reg\[[0-2]\]}]
set bridge_l0_completions_rg_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/completions/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l0_completions_rg_from -to $bridge_l0_completions_rg_to
set_bus_skew 4.000 -from $bridge_l0_completions_rg_from -to $bridge_l0_completions_rg_to

set bridge_l1_commands_wg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/commands/wg_reg\[[0-2]\]}]
set bridge_l1_commands_wg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/commands/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l1_commands_wg_from -to $bridge_l1_commands_wg_to
set_bus_skew 4.000 -from $bridge_l1_commands_wg_from -to $bridge_l1_commands_wg_to

set bridge_l1_commands_rg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/commands/rg_reg\[[0-2]\]}]
set bridge_l1_commands_rg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/commands/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_commands_rg_from -to $bridge_l1_commands_rg_to
set_bus_skew 4.000 -from $bridge_l1_commands_rg_from -to $bridge_l1_commands_rg_to

set bridge_l1_payloads_wg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/payloads/wg_reg\[[0-2]\]}]
set bridge_l1_payloads_wg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/payloads/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l1_payloads_wg_from -to $bridge_l1_payloads_wg_to
set_bus_skew 4.000 -from $bridge_l1_payloads_wg_from -to $bridge_l1_payloads_wg_to

set bridge_l1_payloads_rg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/payloads/rg_reg\[[0-2]\]}]
set bridge_l1_payloads_rg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/payloads/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_payloads_rg_from -to $bridge_l1_payloads_rg_to
set_bus_skew 4.000 -from $bridge_l1_payloads_rg_from -to $bridge_l1_payloads_rg_to

set bridge_l1_completions_wg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/completions/wg_reg\[[0-2]\]}]
set bridge_l1_completions_wg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/completions/ws1_reg\[[0-2]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_completions_wg_from -to $bridge_l1_completions_wg_to
set_bus_skew 4.000 -from $bridge_l1_completions_wg_from -to $bridge_l1_completions_wg_to

set bridge_l1_completions_rg_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/completions/rg_reg\[[0-2]\]}]
set bridge_l1_completions_rg_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/completions/rs1_reg\[[0-2]\]}]
set_max_delay -datapath_only 5.000 -from $bridge_l1_completions_rg_from -to $bridge_l1_completions_rg_to
set_bus_skew 4.000 -from $bridge_l1_completions_rg_from -to $bridge_l1_completions_rg_to

set bridge_l0_fault_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/dma_fault_reg}]
set bridge_l0_fault_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/fault_s1_reg}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_fault_from -to $bridge_l0_fault_to

set bridge_l0_sent_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/sent_generation_reg\[[0-1]\]}]
set bridge_l0_sent_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/sent_s1_reg\[[0-1]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_sent_from -to $bridge_l0_sent_to

set bridge_l1_fault_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/dma_fault_reg}]
set bridge_l1_fault_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/fault_s1_reg}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_fault_from -to $bridge_l1_fault_to

set bridge_l1_sent_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/sent_generation_reg\[[0-1]\]}]
set bridge_l1_sent_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/sent_s1_reg\[[0-1]\]}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_sent_from -to $bridge_l1_sent_to

set bridge_ready_from [get_cells -quiet -hier -regexp {dr_reg\[2\]}]
set bridge_ready_to [get_cells -quiet -hier -regexp {ready_s1_reg}]
set_max_delay -datapath_only 4.000 -from $bridge_ready_from -to $bridge_ready_to

set bridge_l0_stop_from [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/stop_source_reg}]
set bridge_l0_stop_to [get_cells -quiet -hier -regexp {lanes\[0\]\.bridge/stop_d1_reg}]
set_max_delay -datapath_only 5.000 -from $bridge_l0_stop_from -to $bridge_l0_stop_to

set bridge_l1_stop_from [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/stop_source_reg}]
set bridge_l1_stop_to [get_cells -quiet -hier -regexp {lanes\[1\]\.bridge/stop_d1_reg}]
set_max_delay -datapath_only 5.000 -from $bridge_l1_stop_from -to $bridge_l1_stop_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l0_commands_data_to [get_pins -quiet -hier -regexp {lanes\[0\]\.bridge/commands/output_data_reg\[[0-9]+\]/D}]
set bridge_l0_commands_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l0_commands_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_commands_data_from -to $bridge_l0_commands_data_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l0_payloads_data_to [get_pins -quiet -hier -regexp {lanes\[0\]\.bridge/payloads/output_data_reg\[[0-9]+\]/D}]
set bridge_l0_payloads_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l0_payloads_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 4.000 -from $bridge_l0_payloads_data_from -to $bridge_l0_payloads_data_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l0_completions_data_to [get_pins -quiet -hier -regexp {lanes\[0\]\.bridge/completions/output_data_reg\[[0-9]+\]/D}]
set bridge_l0_completions_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l0_completions_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 5.000 -from $bridge_l0_completions_data_from -to $bridge_l0_completions_data_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l1_commands_data_to [get_pins -quiet -hier -regexp {lanes\[1\]\.bridge/commands/output_data_reg\[[0-9]+\]/D}]
set bridge_l1_commands_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l1_commands_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_commands_data_from -to $bridge_l1_commands_data_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l1_payloads_data_to [get_pins -quiet -hier -regexp {lanes\[1\]\.bridge/payloads/output_data_reg\[[0-9]+\]/D}]
set bridge_l1_payloads_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l1_payloads_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 4.000 -from $bridge_l1_payloads_data_from -to $bridge_l1_payloads_data_to

# One destination period; capture is enabled only after two pointer-sync stages.
set bridge_l1_completions_data_to [get_pins -quiet -hier -regexp {lanes\[1\]\.bridge/completions/output_data_reg\[[0-9]+\]/D}]
set bridge_l1_completions_data_from [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects $bridge_l1_completions_data_to] -filter {DIRECTION == OUT}] -filter {REF_NAME == RAMD32 || REF_NAME == RAMS32}]
set_max_delay -datapath_only 5.000 -from $bridge_l1_completions_data_from -to $bridge_l1_completions_data_to
