`timescale 1ns / 1ps

module tb_ntrpu_buffer_pool_allocator;
    parameter integer CHANNELS = 2;
    parameter integer DEDICATED_BANKS = 2;
    parameter integer RESERVE_BANKS = 4;
    parameter integer RESERVE_QUOTA = RESERVE_BANKS;
    parameter integer GENERATION_BITS = 64;
    localparam integer BANKS = CHANNELS * DEDICATED_BANKS + RESERVE_BANKS;
    localparam integer BANK_BITS = BANKS <= 1 ? 1 : $clog2(BANKS);
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, stop = 0;
    reg [CHANNELS-1:0] alloc_valid = 0, return_valid = 0;
    reg [CHANNELS*BANK_BITS-1:0] return_bank = 0;
    reg [CHANNELS*GENERATION_BITS-1:0] return_generation = 0;
    wire [CHANNELS-1:0] alloc_ready, return_ready, return_error;
    wire [BANK_BITS-1:0] alloc_bank;
    wire [GENERATION_BITS-1:0] alloc_generation;
    wire fault, drained;
    wire [BANKS-1:0] used_mask, retired_mask;

    ntrpu_buffer_pool_allocator #(
        .CHANNELS(CHANNELS), .DEDICATED_BANKS(DEDICATED_BANKS),
        .RESERVE_BANKS(RESERVE_BANKS), .RESERVE_QUOTA(RESERVE_QUOTA),
        .GENERATION_BITS(GENERATION_BITS)
    ) dut (.*);

    // Driver ledger uses observed grants only to generate well-formed returns.
    // The independent Python oracle checks every input/output edge afterward.
    integer held_owner [0:BANKS-1];
    reg [GENERATION_BITS-1:0] held_generation [0:BANKS-1];
    reg [CHANNELS-1:0] request_bits, release_bits;
    reg [CHANNELS*BANK_BITS-1:0] release_banks;
    reg [CHANNELS*GENERATION_BITS-1:0] release_generations;
    reg [CHANNELS-1:0] pre_ready, pre_error, pre_return_ready;
    reg [BANK_BITS-1:0] pre_bank;
    reg [GENERATION_BITS-1:0] pre_generation;
    reg pre_fault;
    reg [BANKS-1:0] pre_used, pre_retired;
    integer cycle = 0;
    integer seed, i, p, b, picked, saved_bank, random_step;
    reg [31:0] rng;
    reg [GENERATION_BITS-1:0] saved_generation;

    function [31:0] random_word;
        begin
            rng = rng * 32'd1664525 + 32'd1013904223;
            random_word = rng;
        end
    endfunction

    task step;
        input reset_value;
        input stop_value;
        input [CHANNELS-1:0] requests;
        input [CHANNELS-1:0] releases;
        input [CHANNELS*BANK_BITS-1:0] banks;
        input [CHANNELS*GENERATION_BITS-1:0] generations;
        integer q, bank;
        begin
            @(negedge clk);
            rst_n = reset_value;
            stop = stop_value;
            alloc_valid = requests;
            return_valid = releases;
            return_bank = banks;
            return_generation = generations;
            #1;
            pre_ready = alloc_ready;
            pre_bank = alloc_bank;
            pre_generation = alloc_generation;
            pre_error = return_error;
            pre_return_ready = return_ready;
            pre_fault = rst_n ? fault : 0;
            pre_used = rst_n ? used_mask : 0;
            pre_retired = rst_n ? retired_mask : 0;
            @(posedge clk);
            #1;
            $display("TRACE %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h",
                     cycle, rst_n, stop, alloc_valid, return_valid, return_bank,
                     return_generation, pre_ready, pre_bank, pre_generation,
                     pre_error, pre_return_ready, pre_fault, pre_used, pre_retired,
                     fault, used_mask, retired_mask, drained);
            cycle = cycle + 1;
            if (!rst_n) begin
                for (q = 0; q < BANKS; q = q + 1) begin
                    held_owner[q] = -1;
                    held_generation[q] = 0;
                end
            end else begin
                for (q = 0; q < CHANNELS; q = q + 1) begin
                    bank = return_bank[q*BANK_BITS +: BANK_BITS];
                    if (return_valid[q] && bank < BANKS && held_owner[bank] == q &&
                        held_generation[bank] == return_generation[q*GENERATION_BITS +: GENERATION_BITS])
                        held_owner[bank] = -1;
                end
                for (q = 0; q < CHANNELS; q = q + 1) begin
                    if (pre_ready[q]) begin
                        if (pre_bank >= BANKS || pre_generation == 0)
                            $fatal(1, "invalid grant");
                        held_owner[pre_bank] = q;
                        held_generation[pre_bank] = pre_generation;
                    end
                end
            end
        end
    endtask

    task prepare_returns;
        integer q, bank;
        begin
            release_bits = 0;
            release_banks = 0;
            release_generations = 0;
            for (q = 0; q < CHANNELS; q = q + 1) begin
                for (bank = 0; bank < BANKS; bank = bank + 1) begin
                    if (held_owner[bank] == q && !release_bits[q]) begin
                        release_bits[q] = 1;
                        release_banks[q*BANK_BITS +: BANK_BITS] = bank;
                        release_generations[q*GENERATION_BITS +: GENERATION_BITS] = held_generation[bank];
                    end
                end
            end
        end
    endtask

    task drain_all;
        integer n;
        begin
            for (n = 0; n < BANKS + 1; n = n + 1) begin
                prepare_returns;
                step(1, 1, 0, release_bits, release_banks, release_generations);
            end
            if (!drained) $fatal(1, "drain failed");
        end
    endtask

    task cold_reset;
        begin
            // All simulated producers/returns are quiescent here.
            drain_all;
            step(0, 0, {CHANNELS{1'b1}}, 0, 0, 0);
            step(1, 1, 0, 0, 0, 0);
            if (fault || !drained) $fatal(1, "reset failed");
        end
    endtask

    task allocate_zero;
        begin
            step(1, 0, {{(CHANNELS-1){1'b0}}, 1'b1}, 0, 0, 0);
            if (!pre_ready[0]) $fatal(1, "expected channel zero grant");
            saved_bank = pre_bank;
            saved_generation = pre_generation;
        end
    endtask

    task stale_return;
        begin
            release_banks = 0;
            release_generations = 0;
            release_banks[0 +: BANK_BITS] = saved_bank;
            release_generations[0 +: GENERATION_BITS] = saved_generation;
            step(1, 0, {CHANNELS{1'b1}}, {{(CHANNELS-1){1'b0}}, 1'b1}, release_banks, release_generations);
            if (!fault || pre_ready != 0) $fatal(1, "stale return did not block allocation");
        end
    endtask

    initial begin
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        rng = seed;
        for (b = 0; b < BANKS; b = b + 1) begin
            held_owner[b] = -1;
            held_generation[b] = 0;
        end
        $display("CASE cold_reset");
        step(0, 0, {CHANNELS{1'b1}}, {CHANNELS{1'b1}}, 0, 0);
        $display("CASE simultaneous_requests");
        for (i = 0; i < BANKS + CHANNELS + 2; i = i + 1)
            step(1, 0, {CHANNELS{1'b1}}, 0, 0, 0);
        $display("CASE simultaneous_return_reuse");
        prepare_returns;
        step(1, 0, {CHANNELS{1'b1}}, release_bits, release_banks, release_generations);
        $display("CASE stop_keeps_returns_live");
        prepare_returns;
        step(1, 1, {CHANNELS{1'b1}}, release_bits, release_banks, release_generations);
        if (pre_ready != 0 || fault) $fatal(1, "stop semantics");

        $display("CASE quota");
        cold_reset;
        for (i = 0; i < DEDICATED_BANKS + RESERVE_QUOTA; i = i + 1)
            allocate_zero;
        step(1, 0, {{(CHANNELS-1){1'b0}}, 1'b1}, 0, 0, 0);
        if (pre_ready != 0) $fatal(1, "quota exceeded");

        $display("CASE stale_generation_after_reuse");
        cold_reset;
        allocate_zero;
        release_banks = 0;
        release_generations = 0;
        release_banks[0 +: BANK_BITS] = saved_bank;
        release_generations[0 +: GENERATION_BITS] = saved_generation;
        step(1, 0, {{(CHANNELS-1){1'b0}}, 1'b1}, {{(CHANNELS-1){1'b0}}, 1'b1}, release_banks, release_generations);
        if (!pre_ready[0] || pre_bank != saved_bank || pre_generation != saved_generation + 1'b1)
            $fatal(1, "same-cycle reuse failed");
        stale_return;
        $display("CASE fault_is_sticky_but_drains");
        step(1, 0, {CHANNELS{1'b1}}, 0, 0, 0);
        if (!fault || pre_ready != 0) $fatal(1, "fault not sticky");
        drain_all;

        $display("CASE double_return");
        cold_reset;
        allocate_zero;
        prepare_returns;
        step(1, 0, 0, release_bits, release_banks, release_generations);
        stale_return;

        if (CHANNELS > 1) begin
            $display("CASE wrong_owner");
            cold_reset;
            allocate_zero;
            release_bits = 0;
            release_banks = 0;
            release_generations = 0;
            release_bits[1] = 1;
            release_banks[BANK_BITS +: BANK_BITS] = saved_bank;
            release_generations[GENERATION_BITS +: GENERATION_BITS] = saved_generation;
            step(1, 0, {CHANNELS{1'b1}}, release_bits, release_banks, release_generations);
            if (!fault || !used_mask[saved_bank]) $fatal(1, "foreign owner released bank");

            $display("CASE conflicting_returns");
            cold_reset;
            allocate_zero;
            release_bits = 0;
            release_banks = 0;
            release_generations = 0;
            for (p = 0; p < 2; p = p + 1) begin
                release_bits[p] = 1;
                release_banks[p*BANK_BITS +: BANK_BITS] = saved_bank;
                release_generations[p*GENERATION_BITS +: GENERATION_BITS] = saved_generation;
            end
            step(1, 0, {CHANNELS{1'b1}}, release_bits, release_banks, release_generations);
            if (!fault || !drained) $fatal(1, "conflicting return handling");
        end
        if ((1 << BANK_BITS) > BANKS) begin
            $display("CASE invalid_bank_index");
            cold_reset;
            release_banks = 0;
            release_banks[0 +: BANK_BITS] = BANKS;
            step(1, 0, {CHANNELS{1'b1}}, {{(CHANNELS-1){1'b0}}, 1'b1}, release_banks, 0);
            if (!fault) $fatal(1, "out-of-range bank accepted");
        end
        if (GENERATION_BITS <= 4 && BANKS == 1) begin
            $display("CASE generation_exhaustion");
            cold_reset;
            for (i = 0; i < (1 << GENERATION_BITS) - 1; i = i + 1) begin
                allocate_zero;
                prepare_returns;
                step(1, 0, 0, release_bits, release_banks, release_generations);
            end
            step(1, 0, 1, 0, 0, 0);
            if (pre_ready != 0 || retired_mask != 1 || !drained)
                $fatal(1, "generation wrapped");
        end

        $display("CASE seeded_random");
        cold_reset;
        for (random_step = 0; random_step < 512; random_step = random_step + 1) begin
            request_bits = random_word() >> 16;
            prepare_returns;
            for (p = 0; p < CHANNELS; p = p + 1) begin
                if (((random_word() >> 16) & 3) != 0) release_bits[p] = 0;
            end
            step(1, ((random_word() >> 16) & 15) == 0, request_bits, release_bits, release_banks, release_generations);
            if (fault) $fatal(1, "legal randomized traffic faulted");
        end
        $display("CASE final_drain");
        drain_all;
        $display("PASS seed=%0d cycles=%0d channels=%0d dedicated=%0d reserve=%0d quota=%0d generation=%0d",
                 seed, cycle, CHANNELS, DEDICATED_BANKS, RESERVE_BANKS, RESERVE_QUOTA, GENERATION_BITS);
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "watchdog");
    end
endmodule
