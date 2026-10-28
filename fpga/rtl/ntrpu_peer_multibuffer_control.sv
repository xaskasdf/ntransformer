`timescale 1ns / 1ps

// Isolated return-control experiment; compact baseline remains unchanged.
// Capture the current lease at start_write, before any final W handshake.
// PACKED_RAM=0 and NTRPU_REFERENCE_ALLOCATOR are controlled ablation options.
// Isolated prototype. Input frames are prevalidated and contain exactly LEN+1
// full 512-bit beats. stop closes admission but DRAINS accepted work.
// Not a replacement for the guard's epoch/abort/rearm protocol.
module ntrpu_peer_multibuffer_control #(
    parameter integer PACKED_RAM = 1,
    parameter integer CHANNELS = 2,
    parameter integer RESERVE_BANKS = 4,
    parameter integer BANK_BEATS = 64,
    parameter integer DESCRIPTORS = 16,
    parameter integer MAX_OUTSTANDING = 4,
    parameter integer BANKS = 2 * CHANNELS + RESERVE_BANKS,
    parameter integer BANK_BITS = BANKS <= 1 ? 1 : $clog2(BANKS),
    parameter integer CHANNEL_BITS = CHANNELS <= 1 ? 1 : $clog2(CHANNELS),
    parameter integer COUNT_BITS = $clog2(DESCRIPTORS + 1)
) (
    input wire clk, rst_n, stop,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [CHANNEL_BITS-1:0] cmd_channel,
    input wire [63:0] cmd_addr, cmd_tag,
    input wire [7:0] cmd_len,
    input wire data_valid,
    output wire data_ready,
    input wire [511:0] data,
    output wire [CHANNELS-1:0] m_awvalid,
    input wire [CHANNELS-1:0] m_awready,
    output wire [CHANNELS*64-1:0] m_awaddr,
    output wire [CHANNELS*8-1:0] m_awlen,
    output wire [CHANNELS*4-1:0] m_awid, m_awcache,
    output wire [CHANNELS*3-1:0] m_awsize,
    output wire [CHANNELS*2-1:0] m_awburst,
    output wire [CHANNELS-1:0] m_wvalid, m_wlast,
    input wire [CHANNELS-1:0] m_wready,
    output wire [CHANNELS*512-1:0] m_wdata,
    output wire [CHANNELS*64-1:0] m_wstrb,
    input wire [CHANNELS-1:0] m_bvalid,
    output wire [CHANNELS-1:0] m_bready,
    input wire [CHANNELS*4-1:0] m_bid,
    input wire [CHANNELS*2-1:0] m_bresp,
    output wire [CHANNELS-1:0] cpl_valid,
    input wire [CHANNELS-1:0] cpl_ready,
    output wire [CHANNELS*64-1:0] cpl_tag,
    output wire [CHANNELS*2-1:0] cpl_status,
    output reg fault,
    output wire drained,
    output wire [BANKS-1:0] banks_used,
    output wire [CHANNELS*COUNT_BITS-1:0] pending_counts
);
    localparam integer SLOT_BITS = DESCRIPTORS <= 1 ? 1 : $clog2(DESCRIPTORS);
    localparam integer RAM_BITS = BANK_BEATS <= 1 ? 1 : $clog2(BANK_BEATS);
    localparam [2:0] FREE=0, FILL=1, READY=2, WRITE=3, WAIT_B=4, DONE=5;
    reg [2:0] state [0:CHANNELS*DESCRIPTORS-1];
    reg [63:0] address [0:CHANNELS*DESCRIPTORS-1];
    reg [63:0] tag [0:CHANNELS*DESCRIPTORS-1];
    reg [7:0] length [0:CHANNELS*DESCRIPTORS-1];
    reg [BANK_BITS-1:0] bank [0:CHANNELS*DESCRIPTORS-1];
    reg [63:0] generation [0:CHANNELS*DESCRIPTORS-1];
    reg [1:0] status [0:CHANNELS*DESCRIPTORS-1];
    reg [SLOT_BITS-1:0] tail [0:CHANNELS-1], send_head [0:CHANNELS-1];
    reg [SLOT_BITS-1:0] b_head [0:CHANNELS-1], result_head [0:CHANNELS-1];
    reg [COUNT_BITS-1:0] occupied [0:CHANNELS-1], outstanding [0:CHANNELS-1];
    reg fill_active;
    reg [CHANNEL_BITS-1:0] fill_channel;
    reg [SLOT_BITS-1:0] fill_slot;
    reg [BANK_BITS-1:0] fill_bank;
    reg [8:0] fill_index;
    reg [CHANNELS-1:0] active, aw_done, read_pending;
    reg [BANK_BITS-1:0] active_bank [0:CHANNELS-1];
    reg [63:0] active_generation [0:CHANNELS-1];
    reg [8:0] read_index [0:CHANNELS-1];
    reg pending_last [0:CHANNELS-1];
    reg [1:0] fifo_count [0:CHANNELS-1];
    reg [511:0] fifo0 [0:CHANNELS-1], fifo1 [0:CHANNELS-1];
    reg last0 [0:CHANNELS-1], last1 [0:CHANNELS-1];
    wire [CHANNELS-1:0] start_write, read_issue, pop_w, finish_w, accept_b;
    reg [CHANNELS-1:0] alloc_valid;
    wire [CHANNELS-1:0] alloc_ready, return_ready, return_error;
    wire [BANK_BITS-1:0] alloc_bank;
    wire [63:0] alloc_generation;
    wire [CHANNELS*BANK_BITS-1:0] return_bank;
    wire [CHANNELS*64-1:0] return_generation;
    wire allocator_fault, allocator_drained;
    wire [BANKS-1:0] retired_mask;
    wire [BANKS*512-1:0] ram_q;
    reg [BANKS-1:0] ram_read_enable;
    reg [BANKS*RAM_BITS-1:0] ram_read_address;
    wire [8:0] requested_beats = {1'b0, cmd_len} + 9'd1;
    wire [14:0] page_end = {3'b0, cmd_addr[11:0]} + {requested_beats, 6'b0};
    wire bad_cmd = cmd_valid && !fill_active &&
        (cmd_channel >= CHANNELS || cmd_addr[5:0] != 0 ||
         requested_beats > BANK_BEATS || page_end > 4096);
    wire [CHANNELS-1:0] response_error;
    wire fault_now = allocator_fault || (|return_error) || (|response_error) || bad_cmd;
    wire accept_cmd = cmd_valid && cmd_ready;
    reg descriptors_empty;
    integer p, i, index;

    function [SLOT_BITS-1:0] next_slot;
        input [SLOT_BITS-1:0] current;
        begin next_slot = current == DESCRIPTORS-1 ? 0 : current + 1'b1; end
    endfunction

`ifdef NTRPU_REFERENCE_ALLOCATOR
    ntrpu_buffer_pool_allocator #(
`else
    ntrpu_buffer_pool_allocator_parallel #(
`endif
        .CHANNELS(CHANNELS), .DEDICATED_BANKS(2),
        .RESERVE_BANKS(RESERVE_BANKS)) allocator (
        .clk(clk), .rst_n(rst_n), .stop(stop || fault || fault_now),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready), .alloc_bank(alloc_bank),
        .alloc_generation(alloc_generation), .return_valid(finish_w),
        .return_ready(return_ready), .return_bank(return_bank),
        .return_generation(return_generation), .return_error(return_error),
        .fault(allocator_fault), .used_mask(banks_used), .retired_mask(retired_mask),
        .drained(allocator_drained)
    );

    assign cmd_ready = |alloc_ready;
    assign data_ready = rst_n && fill_active;
    assign drained = allocator_drained && descriptors_empty && !fill_active;
    always @* begin
        alloc_valid = 0;
        if (cmd_channel < CHANNELS && !fill_active && !bad_cmd && cmd_valid)
            if (occupied[cmd_channel] < DESCRIPTORS)
                alloc_valid[cmd_channel] = 1;
        descriptors_empty = 1;
        ram_read_enable = 0;
        ram_read_address = 0;
        for (p = 0; p < CHANNELS; p = p + 1) begin
            if (occupied[p] != 0) descriptors_empty = 0;
            if (read_issue[p]) begin
                ram_read_enable[bank[p*DESCRIPTORS + send_head[p]]] = 1;
                ram_read_address[bank[p*DESCRIPTORS + send_head[p]]*RAM_BITS +: RAM_BITS] = read_index[p];
            end
        end
    end

    genvar g;
    // Logical banks share depth. One complete replica per reader keeps 1R/1W
    // per physical array; ingress writes are broadcast to every replica.
    localparam integer POOL_WORDS = BANKS * BANK_BEATS;
    localparam integer POOL_ADDR_BITS = POOL_WORDS <= 1 ? 1 : $clog2(POOL_WORDS);
    wire [POOL_ADDR_BITS-1:0] pool_write_address = fill_bank * BANK_BEATS + fill_index;
    wire [CHANNELS*512-1:0] channel_ram_q;
    generate if (PACKED_RAM) begin: packed_pool
        for (g=0; g<CHANNELS; g=g+1) begin: reader
            (* ram_style = "block" *) reg [511:0] payload [0:POOL_WORDS-1];
            reg [511:0] q;
            wire [POOL_ADDR_BITS-1:0] pool_read_address =
                bank[g*DESCRIPTORS+send_head[g]] * BANK_BEATS + read_index[g];
            assign channel_ram_q[g*512 +:512] = q;
            always @(posedge clk) begin
                if (rst_n && data_valid && data_ready)
                    payload[pool_write_address] <= data;
                if (rst_n && read_issue[g])
                    q <= payload[pool_read_address];
            end
        end
    end else begin: reference_banks
        for (g=0; g<BANKS; g=g+1) begin: memory_bank
        // A physically separate 1R/1W bank. BRAM inference/resource use not yet measured.
        (* ram_style = "block" *) reg [511:0] payload [0:BANK_BEATS-1];
        reg [511:0] q;
        assign ram_q[g*512 +: 512] = q;
        always @(posedge clk) begin
            if (rst_n && data_valid && data_ready && fill_bank == g)
                payload[fill_index[RAM_BITS-1:0]] <= data;
            if (rst_n && ram_read_enable[g])
                q <= payload[ram_read_address[g*RAM_BITS +: RAM_BITS]];
        end
        end

        for (g=0; g<CHANNELS; g=g+1) begin: select_reader
            assign channel_ram_q[g*512 +:512] = ram_q[bank[g*DESCRIPTORS+send_head[g]]*512 +:512];
        end
    end endgenerate

    generate for (g=0; g<CHANNELS; g=g+1) begin: channel_io
        wire [SLOT_BITS-1:0] sh = send_head[g];
        assign start_write[g] = rst_n && !active[g] && state[g*DESCRIPTORS+sh] == READY &&
                                outstanding[g] < MAX_OUTSTANDING;
        assign m_awvalid[g] = rst_n && active[g] && !aw_done[g];
        assign m_awaddr[g*64 +:64] = address[g*DESCRIPTORS+sh];
        assign m_awlen[g*8 +:8] = length[g*DESCRIPTORS+sh];
        assign m_awid[g*4 +:4] = 0;
        assign m_awcache[g*4 +:4] = 0;
        assign m_awsize[g*3 +:3] = 6;
        assign m_awburst[g*2 +:2] = 1;
        assign m_wvalid[g] = rst_n && active[g] && aw_done[g] && fifo_count[g] != 0;
        assign m_wdata[g*512 +:512] = fifo0[g];
        assign m_wlast[g] = last0[g];
        assign m_wstrb[g*64 +:64] = {64{1'b1}};
        assign pop_w[g] = m_wvalid[g] && m_wready[g];
        assign finish_w[g] = pop_w[g] && last0[g];
        assign read_issue[g] = rst_n && active[g] && read_index[g] <= length[g*DESCRIPTORS+sh] &&
            (int'(fifo_count[g]) + int'(read_pending[g]) - int'(pop_w[g]) < 2);
        assign return_bank[g*BANK_BITS +: BANK_BITS] = active_bank[g];
        assign return_generation[g*64 +:64] = active_generation[g];
        assign m_bready[g] = rst_n && outstanding[g] != 0 && state[g*DESCRIPTORS+b_head[g]] == WAIT_B;
        assign accept_b[g] = m_bvalid[g] && m_bready[g] && m_bid[g*4 +:4] == 0;
        assign response_error[g] = m_bvalid[g] && m_bready[g] &&
            (m_bid[g*4 +:4] != 0 || m_bresp[g*2 +:2] != 0);
        assign cpl_valid[g] = rst_n && occupied[g] != 0 && state[g*DESCRIPTORS+result_head[g]] == DONE;
        assign cpl_tag[g*64 +:64] = tag[g*DESCRIPTORS+result_head[g]];
        assign cpl_status[g*2 +:2] = status[g*DESCRIPTORS+result_head[g]];
        assign pending_counts[g*COUNT_BITS +: COUNT_BITS] = outstanding[g];
    end endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            fault <= 0;
            fill_active <= 0;
            fill_channel <= 0;
            fill_slot <= 0;
            fill_bank <= 0;
            fill_index <= 0;
            active <= 0;
            aw_done <= 0;
            read_pending <= 0;
            for (i=0; i<CHANNELS; i=i+1) begin
                tail[i]<=0; send_head[i]<=0; b_head[i]<=0; result_head[i]<=0;
                active_bank[i]<=0; active_generation[i]<=0;
                occupied[i]<=0; outstanding[i]<=0; read_index[i]<=0;
                fifo_count[i]<=0; pending_last[i]<=0; last0[i]<=0; last1[i]<=0;
                fifo0[i]<=0; fifo1[i]<=0;
            end
            for (i=0; i<CHANNELS*DESCRIPTORS; i=i+1) begin
                state[i]<=FREE; address[i]<=0; tag[i]<=0; length[i]<=0;
                bank[i]<=0; generation[i]<=0; status[i]<=0;
            end
        end else begin
            if (fault_now) fault <= 1;
            if (accept_cmd) begin
                fill_active <= 1;
                fill_channel <= cmd_channel;
                fill_slot <= tail[cmd_channel];
                fill_bank <= alloc_bank;
                fill_index <= 0;
                state[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= FILL;
                address[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= cmd_addr;
                tag[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= cmd_tag;
                length[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= cmd_len;
                bank[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= alloc_bank;
                generation[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= alloc_generation;
                status[cmd_channel*DESCRIPTORS+tail[cmd_channel]] <= 0;
                tail[cmd_channel] <= next_slot(tail[cmd_channel]);
            end
            if (data_valid && data_ready) begin
                if (fill_index == length[fill_channel*DESCRIPTORS+fill_slot]) begin
                    fill_active <= 0;
                    state[fill_channel*DESCRIPTORS+fill_slot] <= READY;
                end else fill_index <= fill_index + 1'b1;
            end
            for (i=0; i<CHANNELS; i=i+1) begin
                case ({accept_cmd && cmd_channel == i, cpl_valid[i] && cpl_ready[i]})
                    2'b10: occupied[i] <= occupied[i]+1'b1;
                    2'b01: occupied[i] <= occupied[i]-1'b1;
                    default: ;
                endcase
                case ({start_write[i], accept_b[i]})
                    2'b10: outstanding[i] <= outstanding[i]+1'b1;
                    2'b01: outstanding[i] <= outstanding[i]-1'b1;
                    default: ;
                endcase
                if (cpl_valid[i] && cpl_ready[i]) begin
                    state[i*DESCRIPTORS+result_head[i]] <= FREE;
                    result_head[i] <= next_slot(result_head[i]);
                end
                if (start_write[i]) begin
                    active_bank[i] <= bank[i*DESCRIPTORS+send_head[i]];
                    active_generation[i] <= generation[i*DESCRIPTORS+send_head[i]];
                    active[i] <= 1;
                    aw_done[i] <= 0;
                    state[i*DESCRIPTORS+send_head[i]] <= WRITE;
                    read_index[i] <= 0;
                end
                if (m_awvalid[i] && m_awready[i]) aw_done[i] <= 1;
                read_pending[i] <= read_issue[i];
                if (read_issue[i]) begin
                    pending_last[i] <= read_index[i] == length[i*DESCRIPTORS+send_head[i]];
                    read_index[i] <= read_index[i] + 1'b1;
                end
                case ({read_pending[i], pop_w[i]})
                    2'b10: begin
                        if (fifo_count[i] == 0) begin
                            fifo0[i] <= channel_ram_q[i*512 +:512];
                            last0[i] <= pending_last[i];
                        end else begin
                            fifo1[i] <= channel_ram_q[i*512 +:512];
                            last1[i] <= pending_last[i];
                        end
                        fifo_count[i] <= fifo_count[i]+1'b1;
                    end
                    2'b01: begin
                        fifo0[i] <= fifo1[i]; last0[i] <= last1[i];
                        fifo_count[i] <= fifo_count[i]-1'b1;
                    end
                    2'b11: begin
                        if (fifo_count[i] == 1) begin
                            fifo0[i] <= channel_ram_q[i*512 +:512];
                            last0[i] <= pending_last[i];
                        end else begin
                            fifo0[i] <= fifo1[i]; last0[i] <= last1[i];
                            fifo1[i] <= channel_ram_q[i*512 +:512];
                            last1[i] <= pending_last[i];
                        end
                    end
                    default: ;
                endcase
                if (finish_w[i]) begin
                    active[i] <= 0;
                    state[i*DESCRIPTORS+send_head[i]] <= WAIT_B;
                    send_head[i] <= next_slot(send_head[i]);
                end
                if (accept_b[i]) begin
                    state[i*DESCRIPTORS+b_head[i]] <= DONE;
                    status[i*DESCRIPTORS+b_head[i]] <= m_bresp[i*2 +:2];
                    b_head[i] <= next_slot(b_head[i]);
                end
            end
        end
    end
`ifndef SYNTHESIS
    initial begin
        if (CHANNELS < 1 || BANK_BEATS < 1 || BANK_BEATS > 64 ||
            DESCRIPTORS < 1 || MAX_OUTSTANDING < 1 || MAX_OUTSTANDING > DESCRIPTORS ||
            BANKS != 2*CHANNELS+RESERVE_BANKS)
            $fatal(1, "invalid multibuffer parameters");
    end
`endif
endmodule
