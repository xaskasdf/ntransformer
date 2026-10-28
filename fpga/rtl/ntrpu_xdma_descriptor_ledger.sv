`timescale 1ns / 1ps
// Descriptor accounting in axi_aclk, channel 0 H2C/C2H. Publication must be
// connected to the actual descriptor visibility/ownership transfer, not intent.
// Status bit 3 is one Descriptor Done event per cycle; bit 6 is Run, bit 0 Busy.
// Host closure must already be a coherent, revocable producer certificate.
// accounted covers descriptor obligations ONLY, never PCIe posted reception,
// BAR, local replies, CDC or global reset permission. Release requires the
// caller's complete global retirement, beyond this module's local predicate.
module ntrpu_xdma_descriptor_ledger #(
    parameter integer COUNTER_BITS = 32
) (
    input wire clk, cold_rst_n, core_ready,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_generation, arm_tag,
    input wire publish_valid, publish_channel,
    input wire [63:0] publish_generation, publish_tag,
    input wire [31:0] publish_index,
    input wire [7:0] h2c_status, c2h_status,
    input wire close_request, host_closed,
    input wire [63:0] host_generation, host_tag,
    input wire release_valid,
    output reg active = 0, closed = 0,
    output reg [63:0] generation = 0, tag = 0,
    output reg [1:0] started = 0,
    output reg [8:0] faults = 0,
    output wire [63:0] published, completed, outstanding,
    output wire accounted
);
    localparam [COUNTER_BITS-1:0] MAX_COUNT = {COUNTER_BITS{1'b1}};
    reg [COUNTER_BITS-1:0] issued [0:1];
    reg [COUNTER_BITS-1:0] retired [0:1];
    reg have_identity = 0, host_seen_low = 0;
    reg [1:0] previous_run = 0;
    wire [1:0] run_level = {c2h_status[6], h2c_status[6]};
    wire [1:0] done = {c2h_status[3], h2c_status[3]};
    wire [1:0] rising_run = run_level & ~previous_run;
    wire engines_idle = !(h2c_status[0] || c2h_status[0] || (|run_level));
    wire fresh = arm_generation != 0 && arm_tag != 0 && (!have_identity ||
        arm_generation > generation || (arm_generation == generation && arm_tag > tag));
    wire current_host = host_closed && host_generation == generation && host_tag == tag;
    wire current_publish = publish_generation == generation && publish_tag == tag;
    wire [1:0] accepted_publication;
    assign arm_ready = cold_rst_n && core_ready && !active && !faults && fresh && engines_idle;
    for (genvar c=0; c<2; c=c+1) begin : channels
        initial begin issued[c]=0; retired[c]=0; end
        assign published[c*32+:32] = 32'(issued[c]);
        assign completed[c*32+:32] = 32'(retired[c]);
        assign outstanding[c*32+:32] = 32'(issued[c]) - 32'(retired[c]);
        assign accepted_publication[c] = active && publish_valid && publish_channel == c &&
            current_publish && publish_index == 32'(issued[c]) && issued[c] != MAX_COUNT;
    end
    // Current levels also gate the result: revocation does not wait for a clock.
    assign accounted = cold_rst_n && core_ready && active && closed && !faults &&
        close_request && current_host && !publish_valid && !(|done) && engines_idle &&
        issued[0] == retired[0] && issued[1] == retired[1];

    always @(posedge clk) begin
        if (!cold_rst_n) begin
            active<=0; closed<=0; generation<=0; tag<=0; started<=0; faults<=0;
            have_identity<=0; host_seen_low<=0; previous_run<=0;
            for (integer c=0;c<2;c=c+1) begin issued[c]<=0; retired[c]<=0; end
        end else begin
            previous_run <= run_level;
            if (arm_valid) begin
                if (active || !fresh) faults[1] <= 1;
                else if (arm_ready) begin
                    active<=1; closed<=0; have_identity<=1;
                    generation<=arm_generation; tag<=arm_tag;
                    started<=0; host_seen_low<=!host_closed;
                    for (integer c=0;c<2;c=c+1) begin issued[c]<=0; retired[c]<=0; end
                end
            end
            if (active) begin
                if (!core_ready) faults[0]<=1;
                if (!host_closed) host_seen_low<=1;
                if (close_request && host_closed && host_seen_low && !closed) begin
                    if (current_host) closed<=1;
                    else faults[1]<=1;
                end
                if (closed && (!close_request || !current_host)) faults[8]<=1;
            end
            if (publish_valid) begin
                if (!active) faults[2]<=1;
                else begin
                    if (!current_publish || publish_index != 32'(issued[publish_channel])) faults[1]<=1;
                    if (closed) faults[5]<=1;
                    if (issued[publish_channel] == MAX_COUNT) faults[6]<=1;
                end
            end
            for (integer c=0;c<2;c=c+1) begin
                if (accepted_publication[c]) issued[c] <= issued[c] + 1'b1;
                if (rising_run[c]) begin
                    if (!active) faults[2]<=1;
                    else if (started[c] || closed || !(issued[c] != 0 || accepted_publication[c])) faults[4]<=1;
                    else started[c]<=1;
                end
                if (done[c]) begin
                    if (!active || !started[c] || rising_run[c] ||
                        !(retired[c] < issued[c] || accepted_publication[c])) faults[3]<=1;
                    else if (retired[c] == MAX_COUNT) faults[6]<=1;
                    else retired[c] <= retired[c] + 1'b1;
                end
            end
            if (release_valid) begin
                if (accounted) begin active<=0; closed<=0; host_seen_low<=0; end
                else faults[7]<=1;
            end
        end
    end
endmodule
