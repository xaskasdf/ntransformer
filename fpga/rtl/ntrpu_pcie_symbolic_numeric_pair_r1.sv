`timescale 1ns / 1ps
// Literal A -> B connection used by the reviewed full and focused benches.
module ntrpu_pcie_symbolic_numeric_pair_r1 (
    input wire clk, cold_rst_n,
    input ntrpu_pcie_symbolic_ced_pkg::control_t in_control,
    input ntrpu_pcie_symbolic_frontend_pkg::s1_record_t in_record,
    output ntrpu_pcie_symbolic_numeric_pkg_r1::b_record_t out_b
);
    import ntrpu_pcie_symbolic_numeric_pkg_r1::*;
    wire a_record_t a_value;
    ntrpu_pcie_symbolic_apply_r1 a_stage(.clk(clk),.cold_rst_n(cold_rst_n),
        .in_control(in_control),.in_record(in_record),.out_a(a_value));
    ntrpu_pcie_symbolic_delivery_r1 b_stage(.clk(clk),.cold_rst_n(cold_rst_n),
        .in_a(a_value),.out_b(out_b));
endmodule
