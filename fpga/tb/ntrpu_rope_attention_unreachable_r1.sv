module ntrpu_rope_table_fp32_r1 (
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_start,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,
    output wire [11:0] input_r_address0,
    output wire input_r_ce0,
    input wire [31:0] input_r_q0,
    output wire [5:0] coefficients_address0,
    output wire coefficients_ce0,
    input wire [31:0] coefficients_q0,
    output wire [11:0] output_r_address0,
    output wire output_r_ce0,
    output wire output_r_we0,
    output wire [31:0] output_r_d0,
    input wire [31:0] heads,
    input wire [31:0] position,
    input wire [31:0] coefficient_position,
    output wire [31:0] ap_return
);

// Test sentinel: any arithmetic start fails; no numerical behavior is supplied.
    assign ap_done=0;
    assign ap_idle=1;
    assign ap_ready=1;
    assign input_r_address0=0;
    assign input_r_ce0=0;
    assign coefficients_address0=0;
    assign coefficients_ce0=0;
    assign output_r_address0=0;
    assign output_r_ce0=0;
    assign output_r_we0=0;
    assign output_r_d0=0;
    assign ap_return=0;
    always @(posedge ap_clk)if(!ap_rst && ap_start)$fatal(1,"ADMISSION_UNEXPECTED_ARITHMETIC_START");
endmodule

module ntrpu_attention_tile_fp32_r1 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        query_address0,
        query_ce0,
        query_q0,
        keys_address0,
        keys_ce0,
        keys_q0,
        values_address0,
        values_ce0,
        values_q0,
        state_input_address0,
        state_input_ce0,
        state_input_q0,
        state_output_address0,
        state_output_ce0,
        state_output_we0,
        state_output_d0,
        output_r_address0,
        output_r_ce0,
        output_r_we0,
        output_r_d0,
        tile_start,
        tile_count,
        query_position,
        layer,
        query_head,
        kv_head,
        sequence_r,
        epoch,
        ap_return
);
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
output  [5:0] query_address0;
output   query_ce0;
input  [31:0] query_q0;
output  [11:0] keys_address0;
output   keys_ce0;
input  [31:0] keys_q0;
output  [11:0] values_address0;
output   values_ce0;
input  [31:0] values_q0;
output  [6:0] state_input_address0;
output   state_input_ce0;
input  [63:0] state_input_q0;
output  [6:0] state_output_address0;
output   state_output_ce0;
output   state_output_we0;
output  [63:0] state_output_d0;
output  [5:0] output_r_address0;
output   output_r_ce0;
output   output_r_we0;
output  [31:0] output_r_d0;
input  [31:0] tile_start;
input  [31:0] tile_count;
input  [31:0] query_position;
input  [31:0] layer;
input  [31:0] query_head;
input  [31:0] kv_head;
input  [63:0] sequence_r;
input  [63:0] epoch;
output  [31:0] ap_return;
// Test sentinel: any arithmetic start fails; no numerical behavior is supplied.
    assign ap_done=0;
    assign ap_idle=1;
    assign ap_ready=1;
    assign query_address0=0;
    assign query_ce0=0;
    assign keys_address0=0;
    assign keys_ce0=0;
    assign values_address0=0;
    assign values_ce0=0;
    assign state_input_address0=0;
    assign state_input_ce0=0;
    assign state_output_address0=0;
    assign state_output_ce0=0;
    assign state_output_we0=0;
    assign state_output_d0=0;
    assign output_r_address0=0;
    assign output_r_ce0=0;
    assign output_r_we0=0;
    assign output_r_d0=0;
    assign ap_return=0;
    always @(posedge ap_clk)if(!ap_rst && ap_start)$fatal(1,"ADMISSION_UNEXPECTED_ARITHMETIC_START");
endmodule
