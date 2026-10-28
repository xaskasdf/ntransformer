# Isolated receiver clock only. No board/CDC/external I/O timing claim.
# Do not suppress unconstrained input/output paths or add false paths.
create_clock -period 5.000 -name peer_clk [get_ports clk]
