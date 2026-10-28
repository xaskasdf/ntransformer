`timescale 1ns / 1ps

// Experimental two-input receiver; does not replace the qualified transport.
// Commands reserve one exclusive bank; per-lane payloads may fill concurrently.
// SERIAL_INPUT=1 is an equal-memory, one-beat-per-cycle control configuration.
// Isolated prototype. Input frames are prevalidated and contain exactly LEN+1
// full 512-bit beats. stop closes admission but DRAINS accepted work.
// Not a replacement for the guard's epoch/abort/rearm protocol.
module ntrpu_dual_ingress #(
    parameter integer SERIAL_INPUT = 0,
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
    output wire [CHANNELS-1:0] reserve_can,
    input wire [CHANNELS-1:0] cmd_valid,
    output wire [CHANNELS-1:0] cmd_ready,
    input wire [CHANNELS*64-1:0] cmd_addr, cmd_tag,
    input wire [CHANNELS*8-1:0] cmd_len,
    input wire [CHANNELS-1:0] data_valid,
    output wire [CHANNELS-1:0] data_ready,
    input wire [CHANNELS*512-1:0] data,
    input wire fence_valid,
    output wire fence_ready,
    input wire [63:0] fence_id,
    output wire fence_done_valid,
    input wire fence_done_ready,
    output wire [63:0] fence_done_id,
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
    reg [CHANNELS-1:0] fill_active;
    reg [SLOT_BITS-1:0] fill_slot [0:CHANNELS-1];
    reg [BANK_BITS-1:0] fill_bank [0:CHANNELS-1];
    reg [63:0] fill_generation [0:CHANNELS-1];
    reg [8:0] fill_index [0:CHANNELS-1];
    reg stopped, fence_pending, serial_turn;
    reg [63:0] held_fence_id;
    wire core_drained;
    wire admission_closed=stopped || stop || fence_pending || fence_valid;
    wire [CHANNELS-1:0] payload_eligible=fill_active & {CHANNELS{rst_n}};
    wire [CHANNELS-1:0] payload_requests=payload_eligible & data_valid;
    wire serial_selected=payload_requests[serial_turn] ? serial_turn : !serial_turn;
    assign fence_ready=rst_n && !stopped && !stop && !fault && !fence_pending;
    assign fence_done_valid=rst_n && fence_pending && core_drained;
    assign fence_done_id=held_fence_id;
    always @(posedge clk) begin
        if(!rst_n) begin stopped<=0;fence_pending<=0;held_fence_id<=0;serial_turn<=0;end
        else begin
            if(stop)stopped<=1;
            if(fence_valid && fence_ready)begin fence_pending<=1;held_fence_id<=fence_id;end
            if(fence_done_valid && fence_done_ready)fence_pending<=0;
            if(|(data_valid & data_ready))serial_turn<=!serial_selected;
        end
    end
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
    wire allocator_fault, allocator_drained, return_busy;
    wire [BANKS-1:0] retired_mask;
    wire [BANKS*512-1:0] ram_q;
    reg [BANKS-1:0] ram_read_enable;
    reg [BANKS*RAM_BITS-1:0] ram_read_address;
    wire [CHANNELS-1:0] bad_command;
    wire bad_cmd=|bad_command;
    wire [CHANNELS-1:0] response_error;
    wire fault_now=allocator_fault || (|response_error) || bad_cmd;
    wire [CHANNELS-1:0] accept_cmd=cmd_valid & cmd_ready;
    for(genvar ch=0;ch<CHANNELS;ch++)begin: validate_command
        wire [8:0] beats={1'b0,cmd_len[ch*8+:8]}+9'd1;
        wire [14:0] page_end={3'b0,cmd_addr[ch*64+:12]}+{beats,6'b0};
        assign bad_command[ch]=cmd_valid[ch] && !fill_active[ch] &&
            (cmd_addr[ch*64+:6]!=0 || beats>BANK_BEATS || page_end>4096);
    end
    reg descriptors_empty;
    integer p, i, index;

    function [SLOT_BITS-1:0] next_slot;
        input [SLOT_BITS-1:0] current;
        begin next_slot = current == DESCRIPTORS-1 ? 0 : current + 1'b1; end
    endfunction

    ntrpu_buffer_pool_allocator_return_cut #(
        .CHANNELS(CHANNELS), .DEDICATED_BANKS(2),
        .RESERVE_BANKS(RESERVE_BANKS)) allocator (
        .clk(clk), .rst_n(rst_n), .stop(admission_closed || fault || fault_now),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready), .alloc_bank(alloc_bank),
        .alloc_generation(alloc_generation), .return_valid(finish_w),
        .return_ready(return_ready), .return_bank(return_bank),
        .return_generation(return_generation), .return_error(return_error), .return_busy(return_busy),
        .fault(allocator_fault), .used_mask(banks_used), .retired_mask(retired_mask),
        .drained(allocator_drained)
    );

    assign cmd_ready=alloc_ready;
    assign data_ready=payload_eligible & (SERIAL_INPUT ? (2'b01<<serial_selected) : 2'b11);
    assign core_drained=allocator_drained && descriptors_empty && !(|fill_active);
    assign drained=core_drained && !fence_pending && !fence_valid && !(|accept_cmd);
    always @* begin
        alloc_valid = 0;
        for(integer ch=0;ch<CHANNELS;ch++)
            if(!fill_active[ch] && !bad_cmd && cmd_valid[ch] && occupied[ch]<DESCRIPTORS)
                alloc_valid[ch]=1;
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

    // Eight exclusive banks: each physical array has one selected writer and
    // one selected reader. Parallel ingress uses different leased banks.
    genvar g;
    wire [CHANNELS*512-1:0] channel_ram_q;
    generate for(g=0;g<BANKS;g=g+1)begin: memory_bank
        (* ram_style="block" *) reg [511:0] payload [0:BANK_BEATS-1];
        reg [511:0] q;
        wire write0=data_valid[0] && data_ready[0] && fill_bank[0]==g;
        wire write1=data_valid[1] && data_ready[1] && fill_bank[1]==g;
        wire write_enable=write0 || write1;
        wire [RAM_BITS-1:0] write_address=write0 ? fill_index[0][RAM_BITS-1:0] : fill_index[1][RAM_BITS-1:0];
        wire [511:0] write_data=write0 ? data[0+:512] : data[512+:512];
        assign ram_q[g*512+:512]=q;
        always @(posedge clk)begin
            if(write_enable)payload[write_address]<=write_data;
            if(rst_n && ram_read_enable[g])q<=payload[ram_read_address[g*RAM_BITS+:RAM_BITS]];
        end
    end
    for(g=0;g<CHANNELS;g=g+1)begin: select_reader
        assign channel_ram_q[g*512+:512]=ram_q[bank[g*DESCRIPTORS+send_head[g]]*512+:512];
    end endgenerate

    generate for (g=0; g<CHANNELS; g=g+1) begin: channel_io
        // Conservative offer: returns alone cannot assert availability.
        // With another free bank, the allocator may prefer a same-edge return.
        // Existing command acceptance consumes the real bank and descriptor.
        wire [BANKS-1:0] offer_free;
        for (genvar ob=0;ob<BANKS;ob=ob+1) begin: offer_bank
            assign offer_free[ob] = !banks_used[ob] && !retired_mask[ob] &&
                ((ob>=2*g && ob<2*g+2) || ob>=2*CHANNELS);
        end
        assign reserve_can[g] = rst_n && !admission_closed && !fault && !allocator_fault &&
            !return_busy && !(|response_error) && !fill_active[g] &&
            occupied[g]<DESCRIPTORS && (|offer_free);
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

            active <= 0;
            aw_done <= 0;
            read_pending <= 0;
            for (i=0; i<CHANNELS; i=i+1) begin
                fill_slot[i]<=0;fill_bank[i]<=0;fill_generation[i]<=0;fill_index[i]<=0;
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
            for(i=0;i<CHANNELS;i=i+1)begin
                if(accept_cmd[i])begin
                    fill_active[i]<=1;fill_slot[i]<=tail[i];fill_bank[i]<=alloc_bank;
                    fill_generation[i]<=alloc_generation;fill_index[i]<=0;
                    state[i*DESCRIPTORS+tail[i]]<=FILL;
                    address[i*DESCRIPTORS+tail[i]]<=cmd_addr[i*64+:64];
                    tag[i*DESCRIPTORS+tail[i]]<=cmd_tag[i*64+:64];
                    length[i*DESCRIPTORS+tail[i]]<=cmd_len[i*8+:8];
                    bank[i*DESCRIPTORS+tail[i]]<=alloc_bank;
                    status[i*DESCRIPTORS+tail[i]]<=0;tail[i]<=next_slot(tail[i]);
                end
                if(data_valid[i] && data_ready[i])begin
                    if(fill_index[i]==length[i*DESCRIPTORS+fill_slot[i]])begin
                        fill_active[i]<=0;
                        generation[i*DESCRIPTORS+fill_slot[i]]<=fill_generation[i];
                        state[i*DESCRIPTORS+fill_slot[i]]<=READY;
                    end else fill_index[i]<=fill_index[i]+1'b1;
                end
                case ({accept_cmd[i], cpl_valid[i] && cpl_ready[i]})
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
        if (CHANNELS != 2 || BANK_BEATS < 1 || BANK_BEATS > 64 ||
            DESCRIPTORS < 1 || MAX_OUTSTANDING < 1 || MAX_OUTSTANDING > DESCRIPTORS ||
            BANKS != 2*CHANNELS+RESERVE_BANKS)
            $fatal(1, "invalid multibuffer parameters");
    end
`endif
endmodule
