# Provisional synchronous parent contract for the compact OOC experiment only.
# The 5 ns clock already exists in the validated synthesis checkpoint.
# These are design budgets, not measured PCIe/HBM/board interface delays.
set_input_delay -clock multibuffer_clk -max 1.000 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_input_delay -clock multibuffer_clk -min 0.200 [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock multibuffer_clk -max 1.000 [get_ports -filter {DIRECTION == OUT}]
set_output_delay -clock multibuffer_clk -min -0.100 [get_ports -filter {DIRECTION == OUT}]
set_clock_uncertainty -setup 0.100 [get_clocks multibuffer_clk]
set_clock_uncertainty -hold 0.050 [get_clocks multibuffer_clk]
# No false paths, multicycle paths, reset exceptions, or board pin assignments.
