`timescale 1ns / 1ps

// Same lease/handshake contract as ntrpu_buffer_pool_allocator. Isolated variant:
// bank-local return validation and one-hot bank grants; unchanged cycle contract.
module ntrpu_buffer_pool_allocator_return_cut #(
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
    output wire [CHANNELS-1:0] alloc_ready,
    output wire [BANK_BITS-1:0] alloc_bank,
    output wire [GENERATION_BITS-1:0] alloc_generation,
    input wire [CHANNELS-1:0] return_valid,
    output wire [CHANNELS-1:0] return_ready,
    input wire [CHANNELS*BANK_BITS-1:0] return_bank,
    input wire [CHANNELS*GENERATION_BITS-1:0] return_generation,
    output wire [CHANNELS-1:0] return_error,
    output wire fault, return_busy,
    output wire [BANKS-1:0] used_mask,
    output wire [BANKS-1:0] retired_mask,
    output wire drained
);
    // A return remains owned through capture and validation. New grants
    // are closed in BOTH phases, before a delayed error can become sticky.
    reg [CHANNELS-1:0] pending;
    reg [CHANNELS*BANK_BITS-1:0] pending_bank;
    reg [CHANNELS*GENERATION_BITS-1:0] pending_generation;
    wire inner_drained;
    assign return_busy = (|return_valid) || (|pending);
    assign return_ready = {CHANNELS{rst_n}};
    assign drained = inner_drained && !(|pending);
    always @(posedge clk) begin
        if (!rst_n) begin
            pending <= 0; pending_bank <= 0; pending_generation <= 0;
        end else begin
            pending <= return_valid;
            for (integer p=0; p<CHANNELS; p=p+1) begin
                if (return_valid[p]) begin
                    pending_bank[p*BANK_BITS+:BANK_BITS] <= return_bank[p*BANK_BITS+:BANK_BITS];
                    pending_generation[p*GENERATION_BITS+:GENERATION_BITS] <= return_generation[p*GENERATION_BITS+:GENERATION_BITS];
                end
            end
        end
    end
    ntrpu_buffer_pool_allocator_cut_core #(
        .CHANNELS(CHANNELS), .DEDICATED_BANKS(DEDICATED_BANKS),
        .RESERVE_BANKS(RESERVE_BANKS), .RESERVE_QUOTA(RESERVE_QUOTA),
        .GENERATION_BITS(GENERATION_BITS), .BANKS(BANKS), .BANK_BITS(BANK_BITS)
    ) core (
        .clk(clk), .rst_n(rst_n), .stop(stop || return_busy),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready), .alloc_bank(alloc_bank),
        .alloc_generation(alloc_generation), .return_valid(pending), .return_ready(),
        .return_bank(pending_bank), .return_generation(pending_generation),
        .return_error(return_error), .fault(fault), .used_mask(used_mask),
        .retired_mask(retired_mask), .drained(inner_drained)
    );
endmodule

`timescale 1ns / 1ps

// Same lease/handshake contract as ntrpu_buffer_pool_allocator. Isolated variant:
// bank-local return validation and one-hot bank grants; unchanged cycle contract.
module ntrpu_buffer_pool_allocator_cut_core #(
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
    output wire [BANK_BITS-1:0] alloc_bank,
    output wire [GENERATION_BITS-1:0] alloc_generation,
    input wire [CHANNELS-1:0] return_valid,
    output wire [CHANNELS-1:0] return_ready,
    input wire [CHANNELS*BANK_BITS-1:0] return_bank,
    input wire [CHANNELS*GENERATION_BITS-1:0] return_generation,
    output wire [CHANNELS-1:0] return_error,
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
    wire [BANKS-1:0] returned;
    wire [CHANNELS*BANKS-1:0] return_match;
    wire [BANKS-1:0] grant;
    wire [BANKS-1:0] available;
    wire [CHANNELS-1:0] candidate_valid;
    wire [CHANNELS*BANKS-1:0] candidate_mask;
    reg [CHANNEL_BITS-1:0] selected_channel;
    reg selected;
    integer offset, channel;

    assign used_mask = used;
    assign drained = ~|used;
    assign return_ready = {CHANNELS{rst_n}};

    // Each comparator sees one fixed bank: no generation/owner mux on return.
    genvar port_index, bank_index;
    generate for (port_index=0; port_index<CHANNELS; port_index=port_index+1) begin: return_port
        for (bank_index=0; bank_index<BANKS; bank_index=bank_index+1) begin: bank_compare
            assign return_match[port_index*BANKS+bank_index] =
                rst_n && return_valid[port_index] &&
                return_bank[port_index*BANK_BITS +: BANK_BITS] == bank_index &&
                used[bank_index] && owner[bank_index] == port_index &&
                generation[bank_index] == return_generation[port_index*GENERATION_BITS +: GENERATION_BITS];
        end
        // An out-of-range bank cannot match, just like an idle, foreign or stale lease.
        assign return_error[port_index] = rst_n && return_valid[port_index] &&
            !(|return_match[port_index*BANKS +: BANKS]);
    end endgenerate

    genvar g;
    generate for (g=0; g<BANKS; g=g+1) begin: bank_state
        wire [CHANNELS-1:0] bank_returns;
        wire [CHANNELS-1:0] bank_grants;
        for (port_index=0; port_index<CHANNELS; port_index=port_index+1) begin: channel_decode
            assign bank_returns[port_index] = return_match[port_index*BANKS+g];
            assign bank_grants[port_index] = alloc_ready[port_index] && candidate_mask[port_index*BANKS+g];
        end
        assign returned[g] = |bank_returns;
        assign grant[g] = |bank_grants;
        assign retired_mask[g] = !used[g] && generation[g] == MAX_GENERATION;
        assign available[g] = !used[g] && generation[g] != MAX_GENERATION;
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
                if (grant[g]) begin
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
        reg [BANKS-1:0] choice;
        integer b;
        always @* begin
            loans = 0;
            found = 0;
            choice = 0;
            for (b=RESERVE_START; b<BANKS; b=b+1)
                if (used[b] && owner[b] == g)
                    loans = loans + 1'b1;
            for (b=g*DEDICATED_BANKS; b<(g+1)*DEDICATED_BANKS; b=b+1) begin
                if (!found && available[b]) begin
                    found = 1;
                    choice[b] = 1'b1;
                end
            end
            if (loans < RESERVE_QUOTA) begin
                for (b=RESERVE_START; b<BANKS; b=b+1) begin
                    if (!found && available[b]) begin
                        found = 1;
                        choice[b] = 1'b1;
                    end
                end
            end
        end
        assign candidate_valid[g] = found;
        assign candidate_mask[g*BANKS +: BANKS] = choice;
    end endgenerate

    always @* begin
        alloc_ready = 0;
        selected = 0;
        selected_channel = 0;
        channel = 0;
        if (rst_n && !stop && !fault && !(|return_valid)) begin
            for (offset=0; offset<CHANNELS; offset=offset+1) begin
                channel = next_channel + offset;
                if (channel >= CHANNELS) channel = channel - CHANNELS;
                if (!selected && alloc_valid[channel] && candidate_valid[channel]) begin
                    selected = 1;
                    selected_channel = channel;
                    alloc_ready[channel] = 1;
                end
            end
        end
    end

    // Encode only the public outputs. Bank state is enabled directly by grant[g].
    // Reductions allow a tree instead of a priority mux over already one-hot terms.
    genvar bit_index;
    generate for (bit_index=0; bit_index<BANK_BITS; bit_index=bit_index+1) begin: bank_encoding
        wire [BANKS-1:0] terms;
        for (bank_index=0; bank_index<BANKS; bank_index=bank_index+1) begin: bank_term
            assign terms[bank_index] = grant[bank_index] && ((bank_index >> bit_index) & 1);
        end
        assign alloc_bank[bit_index] = |terms;
    end endgenerate
    generate for (bit_index=0; bit_index<GENERATION_BITS; bit_index=bit_index+1) begin: generation_selection
        wire [BANKS-1:0] terms;
        for (bank_index=0; bank_index<BANKS; bank_index=bank_index+1) begin: bank_term
            assign terms[bank_index] = grant[bank_index] && next_generation[bank_index][bit_index];
        end
        assign alloc_generation[bit_index] = |terms;
    end endgenerate

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
