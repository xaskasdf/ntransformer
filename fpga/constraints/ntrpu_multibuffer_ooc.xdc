# Isolated synchronous prototype; external I/O delays and CDC remain unqualified.
create_clock -period 5.000 -name multibuffer_clk [get_ports clk]
