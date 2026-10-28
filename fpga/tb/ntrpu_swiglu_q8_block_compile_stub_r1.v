// Compile-only signature from qualified HLS. Never simulate or synthesize.
module ntrpu_swiglu_q8_block (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        gate_address0,
        gate_ce0,
        gate_q0,
        up_address0,
        up_ce0,
        up_q0,
        record_r_address0,
        record_r_ce0,
        record_r_we0,
        record_r_d0,
        elements,
        ap_return
);
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [4:0] gate_address0;
output   gate_ce0;
input  [63:0] gate_q0;
output  [4:0] up_address0;
output   up_ce0;
input  [63:0] up_q0;
output  [2:0] record_r_address0;
output   record_r_ce0;
output   record_r_we0;
output  [63:0] record_r_d0;
input  [31:0] elements;
output  [31:0] ap_return;
endmodule
