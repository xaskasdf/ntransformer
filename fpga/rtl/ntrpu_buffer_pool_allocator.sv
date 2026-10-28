`timescale 1ns / 1ps

// Isolated lease allocator: no payload RAM, AXI, PCIe or descriptor credits.
// Caller returns a lease ONLY after its payload no longer needs the bank.
// Synchronous cold reset requires all producers/returns quiesced and drained.
module ntrpu_buffer_pool_allocator #(
    parameter integer CHANNELS = 2,
    parameter integer DEDICATED_BANKS = 2,
    parameter integer RESERVE_BANKS = 4,
    parameter integer RESERVE_QUOTA = RESERVE_BANKS,
    parameter integer GENERATION_BITS = 64,
    parameter integer BANKS = CHANNELS * DEDICATED_BANKS + RESERVE_BANKS,
    parameter integer BANK_BITS = BANKS <= 1 ? 1 : $clog2(BANKS)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire stop,
    input  wire [CHANNELS-1:0] alloc_valid,
    output reg  [CHANNELS-1:0] alloc_ready,
    output reg  [BANK_BITS-1:0] alloc_bank,
    output reg  [GENERATION_BITS-1:0] alloc_generation,
    // One return per channel per cycle; the port identifies the owner.
    input  wire [CHANNELS-1:0] return_valid,
    output wire [CHANNELS-1:0] return_ready,
    input  wire [CHANNELS*BANK_BITS-1:0] return_bank,
    input  wire [CHANNELS*GENERATION_BITS-1:0] return_generation,
    output reg  [CHANNELS-1:0] return_error,
    output reg  fault,
    output wire [BANKS-1:0] used_mask,
    output reg  [BANKS-1:0] retired_mask,
    output wire drained
);
    localparam integer CHANNEL_BITS = CHANNELS <= 1 ? 1 : $clog2(CHANNELS);
    localparam integer RESERVE_START = CHANNELS * DEDICATED_BANKS;
    localparam [GENERATION_BITS-1:0] MAX_GENERATION = {GENERATION_BITS{1'b1}};
    reg [BANKS-1:0] used;
    reg [CHANNEL_BITS-1:0] owner [0:BANKS-1];
    reg [GENERATION_BITS-1:0] generation [0:BANKS-1];
    reg [CHANNEL_BITS-1:0] next_channel;
    reg [BANKS-1:0] returned;
    reg [BANKS-1:0] available;
    integer loans [0:CHANNELS-1];
    integer p, b, offset, channel, candidate, selected_channel;
    integer seq_bank;
    reg selected;

    assign used_mask = used;
    assign drained = ~|used;
    assign return_ready = {CHANNELS{rst_n}};

    // Grant is a request handshake (one-hot READY), not a queued response.
    // External descriptor/result credits must gate each request beforehand.
    always @* begin
        returned = 0;
        return_error = 0;
        retired_mask = 0;
        available = 0;
        alloc_ready = 0;
        alloc_bank = 0;
        alloc_generation = 0;
        selected = 0;
        selected_channel = 0;
        channel = 0;
        candidate = -1;
        for (p = 0; p < CHANNELS; p = p + 1) begin
            loans[p] = 0;
            if (rst_n && return_valid[p]) begin
                // Check the range before indexing state, including non-powers of two.
                if (return_bank[p*BANK_BITS +: BANK_BITS] >= BANKS) begin
                    return_error[p] = 1;
                end else if (!used[return_bank[p*BANK_BITS +: BANK_BITS]]) begin
                    return_error[p] = 1;
                end else if (owner[return_bank[p*BANK_BITS +: BANK_BITS]] != p ||
                             generation[return_bank[p*BANK_BITS +: BANK_BITS]] !=
                             return_generation[p*GENERATION_BITS +: GENERATION_BITS]) begin
                    return_error[p] = 1;
                end else begin
                    returned[return_bank[p*BANK_BITS +: BANK_BITS]] = 1;
                end
            end
        end
        for (b = 0; b < BANKS; b = b + 1) begin
            retired_mask[b] = !used[b] && generation[b] == MAX_GENERATION;
            available[b] = (!used[b] || returned[b]) && generation[b] != MAX_GENERATION;
            if (b >= RESERVE_START && used[b] && !returned[b])
                loans[owner[b]] = loans[owner[b]] + 1;
        end
        // Round robin between eligible channels; prefer their own banks.
        // Invalid returns suppress allocation on this very edge.
        if (rst_n && !stop && !fault && !(|return_error)) begin
            for (offset = 0; offset < CHANNELS; offset = offset + 1) begin
                channel = next_channel + offset;
                if (channel >= CHANNELS)
                    channel = channel - CHANNELS;
                candidate = -1;
                for (b = 0; b < BANKS; b = b + 1) begin
                    if (b >= channel * DEDICATED_BANKS &&
                        b < (channel + 1) * DEDICATED_BANKS && available[b] && candidate == -1)
                        candidate = b;
                end
                if (candidate == -1 && loans[channel] < RESERVE_QUOTA) begin
                    for (b = RESERVE_START; b < BANKS; b = b + 1) begin
                        if (available[b] && candidate == -1)
                            candidate = b;
                    end
                end
                if (!selected && alloc_valid[channel] && candidate != -1) begin
                    selected = 1;
                    selected_channel = channel;
                    alloc_ready[channel] = 1;
                    alloc_bank = candidate;
                    alloc_generation = generation[candidate] + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            used <= 0;
            fault <= 0;
            next_channel <= 0;
            for (seq_bank = 0; seq_bank < BANKS; seq_bank = seq_bank + 1) begin
                owner[seq_bank] <= 0;
                generation[seq_bank] <= 0;
            end
        end else begin
            if (|return_error)
                fault <= 1;
            for (seq_bank = 0; seq_bank < BANKS; seq_bank = seq_bank + 1) begin
                if (returned[seq_bank])
                    used[seq_bank] <= 0;
            end
            // Later assignment wins for a return+reallocation of the same bank.
            if (|alloc_ready) begin
                used[alloc_bank] <= 1;
                owner[alloc_bank] <= selected_channel;
                generation[alloc_bank] <= alloc_generation;
                if (selected_channel == CHANNELS - 1)
                    next_channel <= 0;
                else
                    next_channel <= selected_channel + 1;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (CHANNELS < 1 || DEDICATED_BANKS < 0 || RESERVE_BANKS < 0 ||
            RESERVE_QUOTA < 0 || RESERVE_QUOTA > RESERVE_BANKS || GENERATION_BITS < 1 ||
            BANKS != CHANNELS * DEDICATED_BANKS + RESERVE_BANKS || BANKS < 1 ||
            BANK_BITS != (BANKS <= 1 ? 1 : $clog2(BANKS)))
            $fatal(1, "invalid allocator parameters");
    end
`endif
endmodule
