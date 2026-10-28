`timescale 1ns / 1ps
module tb_ntrpu_pcie_effect_prefix_order_mutant;
    ntrpu_pcie_inventory_scheduler dut (
        .clk(1'b0),.cold_rst_n(1'b0),.sequence_in(32'b0),
        .profile_valid(1'b0),.user_reset(1'b0),.link_up(1'b0),
        .requester_id(16'b0),.attribute_mask(2'b0),
        .rq_valid(1'b0),.rc_valid(1'b0),.rc_ready(1'b0),
        .rq_ready(4'b0),.rq_data(512'b0),.rc_data(512'b0),
        .rq_user(137'b0),.rc_user(161'b0)
    );
    reg [24:0] actual;
    initial begin
        $display("EFFECT_PREFIX_ORDER_MUTANT_STARTED");
        #1;
        actual=dut.rc_prefix_state(13'd10,12'd1,4'b0011,{22'b0,11'd1,11'd0});
        if (actual!==25'h000000b)
            $fatal(1,"EFFECT_PREFIX_ORDER_MISMATCH actual=%h expected=000000b",actual);
        $fatal(1,"EFFECT_PREFIX_MUTANT_NOT_REJECTED");
    end
endmodule
