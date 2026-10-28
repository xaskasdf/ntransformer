// Diagnostic topology only; deliberately contains no RAM or application logic.
module ntrpu_reset_link_probe (
    input wire clk, dma_clk, cold_reset_n, dma_link_up,
    output wire reader_ready, dma_ready, link_seen
);
    (* ASYNC_REG = "TRUE" *) reg [2:0] sr = 0, dr = 0;
    (* ASYNC_REG = "TRUE" *) reg link_s1 = 0, link_s2 = 0;
    always @(posedge clk or negedge cold_reset_n)
        if (!cold_reset_n) sr <= 0; else sr <= {sr[1:0], 1'b1};
    always @(posedge dma_clk or negedge cold_reset_n)
        if (!cold_reset_n) dr <= 0; else dr <= {dr[1:0], 1'b1};
    always @(posedge clk or negedge cold_reset_n)
        if (!cold_reset_n) begin link_s1 <= 0; link_s2 <= 0; end
        else begin link_s1 <= dma_link_up; link_s2 <= link_s1; end
    assign reader_ready = sr[2];
    assign dma_ready = dr[2];
    assign link_seen = link_s2;
endmodule
