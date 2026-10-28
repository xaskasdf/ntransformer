`timescale 1ns/1ps
`default_nettype none
// Board startup only. PERST and loss of LOCKED must never clear accepted work.
// Supervisor assertion is an explicit cold epoch boundary, including clock stop.
module ntrpu_hbm_board_startup #(
    parameter integer RESET_HOLD_CYCLES = 16,
    parameter integer LOCK_STABLE_CYCLES = 32
)(
    input wire mgmt_clk, supervisor_rst_n, fabric_locked_async,
    output reg fabric_clock_reset = 1,
    output reg configuration_rst_n = 0,
    output reg lock_fault = 0
);
    localparam integer HW = RESET_HOLD_CYCLES < 2 ? 1 : $clog2(RESET_HOLD_CYCLES);
    localparam integer LW = LOCK_STABLE_CYCLES < 2 ? 1 : $clog2(LOCK_STABLE_CYCLES);
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] release_sync = 0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [2:0] locked_sync = 0;
    reg [HW-1:0] hold_count = 0;
    reg [LW-1:0] lock_count = 0;
    initial begin
        if (RESET_HOLD_CYCLES < 1 || LOCK_STABLE_CYCLES < 1)
            $error("Startup cycle counts must be positive");
    end
    always @(posedge mgmt_clk or negedge supervisor_rst_n) begin
        if (!supervisor_rst_n) release_sync <= 0;
        else release_sync <= {release_sync[1:0], 1'b1};
    end
    wire sequencer_rst_n = release_sync[2];
    always @(posedge mgmt_clk or negedge sequencer_rst_n) begin
        if (!sequencer_rst_n) locked_sync <= 0;
        else locked_sync <= {locked_sync[1:0], fabric_locked_async};
    end
    always @(posedge mgmt_clk or negedge sequencer_rst_n) begin
        if (!sequencer_rst_n) begin
            fabric_clock_reset <= 1;
            configuration_rst_n <= 0;
            lock_fault <= 0;
            hold_count <= 0;
            lock_count <= 0;
        end else if (fabric_clock_reset) begin
            if (hold_count == RESET_HOLD_CYCLES-1) fabric_clock_reset <= 0;
            else hold_count <= hold_count + 1'b1;
        end else if (!configuration_rst_n) begin
            if (!locked_sync[2]) lock_count <= 0;
            else if (lock_count == LOCK_STABLE_CYCLES-1) configuration_rst_n <= 1;
            else lock_count <= lock_count + 1'b1;
        end else if (!locked_sync[2]) lock_fault <= 1;
    end
endmodule
`default_nettype wire
