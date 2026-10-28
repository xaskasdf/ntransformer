`timescale 1ns / 1ps

// Same lease/handshake contract as ntrpu_buffer_pool_allocator. Isolated variant:
// static per-channel selection, narrow loan counts, local generation increments.
module ntrpu_buffer_pool_allocator_parallel #(
    parameter integer CHANNELS = 2,
    parameter integer DEDICATED_BANKS = 2,
    parameter integer RESERVE_BANKS = 4,
    parameter integer RESERVE_QUOTA = RESERVE_BANKS,
    parameter integer GENERATION_BITS = 64,
    parameter integer BANKS = CHANNELS * DEDICATED_BANKS + RESERVE_BANKS,
    parameter integer BANK_BITS = BANKS <= 1 ? 1 : $clog2(BANKS)
) (
    input wire clk, rst_n, stop,
    input wire [CHANNELS-1:0] alloc_valid,
    output reg [CHANNELS-1:0] alloc_ready,
    output reg [BANK_BITS-1:0] alloc_bank,
    output reg [GENERATION_BITS-1:0] alloc_generation,
    input wire [CHANNELS-1:0] return_valid,
    output wire [CHANNELS-1:0] return_ready,
    input wire [CHANNELS*BANK_BITS-1:0] return_bank,
    input wire [CHANNELS*GENERATION_BITS-1:0] return_generation,
    output reg [CHANNELS-1:0] return_error,
    output reg fault,
    output wire [BANKS-1:0] used_mask,
    output wire [BANKS-1:0] retired_mask,
    output wire drained
);
    localparam integer CHANNEL_BITS = CHANNELS <= 1 ? 1 : $clog2(CHANNELS);
    localparam integer LOAN_BITS = RESERVE_BANKS < 1 ? 1 : $clog2(RESERVE_BANKS+1);
    localparam integer RESERVE_START = CHANNELS * DEDICATED_BANKS;
    localparam [GENERATION_BITS-1:0] MAX_GENERATION = {GENERATION_BITS{1'b1}};
    reg [BANKS-1:0] used;
    reg [CHANNEL_BITS-1:0] owner [0:BANKS-1];
    reg [GENERATION_BITS-1:0] generation [0:BANKS-1];
    wire [GENERATION_BITS-1:0] next_generation [0:BANKS-1];
    reg [CHANNEL_BITS-1:0] next_channel;
    reg [BANKS-1:0] returned;
    wire [BANKS-1:0] available;
    wire [CHANNELS-1:0] candidate_valid;
    wire [CHANNELS*BANK_BITS-1:0] candidate_bank;
    reg [CHANNEL_BITS-1:0] selected_channel;
    reg selected;
    integer p, offset, channel, mux_bank;

    assign used_mask = used;
    assign drained = ~|used;
    assign return_ready = {CHANNELS{rst_n}};

    always @* begin
        returned = 0;
        return_error = 0;
        for (p=0; p<CHANNELS; p=p+1) begin
            if (rst_n && return_valid[p]) begin
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
    end

    genvar g;
    generate for (g=0; g<BANKS; g=g+1) begin: bank_state
        assign retired_mask[g] = !used[g] && generation[g] == MAX_GENERATION;
        assign available[g] = (!used[g] || returned[g]) && generation[g] != MAX_GENERATION;
        assign next_generation[g] = generation[g] + 1'b1;
        // The selected-bank mux is absent from this counter's D input.
        always @(posedge clk) begin
            if (!rst_n) begin
                used[g] <= 0;
                owner[g] <= 0;
                generation[g] <= 0;
            end else begin
                if (returned[g]) used[g] <= 0;
                // Allocation wins over return on the same edge.
                if ((|alloc_ready) && alloc_bank == g) begin
                    used[g] <= 1;
                    owner[g] <= selected_channel;
                    generation[g] <= next_generation[g];
                end
            end
        end
    end endgenerate

    generate for (g=0; g<CHANNELS; g=g+1) begin: channel_choice
        reg [LOAN_BITS-1:0] loans;
        reg found;
        reg [BANK_BITS-1:0] choice;
        integer b;
        always @* begin
            loans = 0;
            found = 0;
            choice = 0;
            for (b=RESERVE_START; b<BANKS; b=b+1)
                if (used[b] && !returned[b] && owner[b] == g)
                    loans = loans + 1'b1;
            for (b=g*DEDICATED_BANKS; b<(g+1)*DEDICATED_BANKS; b=b+1) begin
                if (!found && available[b]) begin
                    found = 1;
                    choice = b;
                end
            end
            if (loans < RESERVE_QUOTA) begin
                for (b=RESERVE_START; b<BANKS; b=b+1) begin
                    if (!found && available[b]) begin
                        found = 1;
                        choice = b;
                    end
                end
            end
        end
        assign candidate_valid[g] = found;
        assign candidate_bank[g*BANK_BITS +: BANK_BITS] = choice;
    end endgenerate

    always @* begin
        alloc_ready = 0;
        alloc_bank = 0;
        alloc_generation = 0;
        selected = 0;
        selected_channel = 0;
        channel = 0;
        if (rst_n && !stop && !fault && !(|return_error)) begin
            for (offset=0; offset<CHANNELS; offset=offset+1) begin
                channel = next_channel + offset;
                if (channel >= CHANNELS) channel = channel - CHANNELS;
                if (!selected && alloc_valid[channel] && candidate_valid[channel]) begin
                    selected = 1;
                    selected_channel = channel;
                    alloc_ready[channel] = 1;
                    alloc_bank = candidate_bank[channel*BANK_BITS +: BANK_BITS];
                end
            end
        end
        // Select a locally incremented value, rather than add after the mux.
        for (mux_bank=0; mux_bank<BANKS; mux_bank=mux_bank+1)
            if (selected && alloc_bank == mux_bank)
                alloc_generation = next_generation[mux_bank];
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            fault <= 0;
            next_channel <= 0;
        end else begin
            if (|return_error) fault <= 1;
            if (|alloc_ready) begin
                if (selected_channel == CHANNELS-1) next_channel <= 0;
                else next_channel <= selected_channel + 1'b1;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (CHANNELS < 1 || DEDICATED_BANKS < 0 || RESERVE_BANKS < 0 ||
            RESERVE_QUOTA < 0 || RESERVE_QUOTA > RESERVE_BANKS || GENERATION_BITS < 1 ||
            BANKS != CHANNELS*DEDICATED_BANKS+RESERVE_BANKS || BANKS < 1 ||
            BANK_BITS != (BANKS <= 1 ? 1 : $clog2(BANKS)))
            $fatal(1, "invalid allocator parameters");
    end
`endif
endmodule
