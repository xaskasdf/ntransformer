`timescale 1ns / 1ps
// Isolated metadata primitive, not a PCIe protocol/inventory monitor.
// Four registered lookups and six ordered, already-computed updates per cycle.
// Bits: [12:0] remaining, [24:13] address, [27:25] TC, [29:28] attributes.
// A partial update preserves TC/attributes. No allocation bitmap or payload state.
// Controls/tags must be known binary values from the upstream event classifier.
module ntrpu_pcie_tag_metadata (
    input wire clk, cold_rst_n,
    input wire [3:0] lookup_valid,
    input wire [31:0] lookup_tags,
    // Query q sees the pre-edge table plus writes [0, cutoff[q]). 7 is invalid.
    input wire [11:0] lookup_cutoffs,
    input wire [5:0] write_valid, write_full,
    input wire [47:0] write_tags,
    input wire [179:0] write_data,
    output reg [3:0] result_valid=0, result_error=0,
    output reg [119:0] result_data=0
);
    reg [29:0] metadata [0:255];
    reg [119:0] selected_data;
    reg [3:0] selected_valid, selected_error;
    integer q, w;
    always @* begin
        selected_data=0;
        selected_valid=0;
        selected_error=0;
        for (q=0;q<4;q=q+1) begin
            if (lookup_valid[q]) begin
                if (lookup_cutoffs[q*3+:3]>6) selected_error[q]=1;
                else begin
                    selected_valid[q]=1;
                    selected_data[q*30+:30]=metadata[lookup_tags[q*8+:8]];
                    for (w=0;w<6;w=w+1) begin
                        if (w<lookup_cutoffs[q*3+:3] && write_valid[w] &&
                            write_tags[w*8+:8]==lookup_tags[q*8+:8]) begin
                            selected_data[q*30+:25]=write_data[w*30+:25];
                            if (write_full[w])
                                selected_data[q*30+25+:5]=write_data[w*30+25+:5];
                        end
                    end
                end
            end
        end
    end

    integer index, slot;
    always @(posedge clk) begin
        if (!cold_rst_n) begin
            result_valid<=0; result_error<=0; result_data<=0;
            for (index=0;index<256;index=index+1) metadata[index]<=0;
        end else begin
            result_valid<=selected_valid;
            result_error<=selected_error;
            result_data<=selected_data;
            for (index=0;index<256;index=index+1) begin
                // Later matching records win independently for the two fields.
                for (slot=0;slot<6;slot=slot+1) begin
                    if (write_valid[slot] && write_tags[slot*8+:8]==index) begin
                        metadata[index][24:0]<=write_data[slot*30+:25];
                        if (write_full[slot])
                            metadata[index][29:25]<=write_data[slot*30+25+:5];
                    end
                end
            end
        end
    end
endmodule
