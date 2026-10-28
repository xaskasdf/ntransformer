`timescale 1ns/1ps
// Registered-head variant: cached metadata is not a bank/descriptor reservation.
// Actual R4 command acceptance remains the atomic resource handshake.
// The dispatcher's later command phase acknowledges the captured lease only;
// it must NEVER issue a second R4 command. Payload is counted on accepted beats.
// Same clock/reset epoch; full-window/address authorization remains upstream.
// stop closes publication and drains queued jobs. fault quarantines unreserved
// jobs, while already reserved payload and exposed AXI traffic keep draining.
module ntrpu_registered_dispatch (
    input wire clk,rst_n,stop,fault_in,
    input wire [1:0] submit_valid,
    output wire [1:0] submit_ready,
    input wire [463:0] submit_desc,
    output wire fetch_valid,fetch_channel,
    input wire fetch_ready,
    output wire [231:0] fetch_desc,
    input wire payload_valid,
    output wire payload_ready,
    input wire [511:0] payload_data,
    input wire fence_valid,
    output wire fence_ready,
    input wire [63:0] fence_id,
    output wire fence_done_valid,
    input wire fence_done_ready,
    output wire [63:0] fence_done_id,
    output wire [1:0] cpl_valid,
    input wire [1:0] cpl_ready,
    output wire [127:0] cpl_tag,
    output wire [3:0] cpl_status,
    output wire fault,drained,
    output wire [7:0] banks_used,queued,
    output wire [9:0] pending_counts,
    output wire [1:0] m_awvalid,m_wvalid,m_wlast,m_bready,
    input wire [1:0] m_awready,m_wready,m_bvalid,
    output wire [127:0] m_awaddr,m_wstrb,
    output wire [15:0] m_awlen,
    output wire [7:0] m_awid,m_awcache,
    output wire [5:0] m_awsize,
    output wire [3:0] m_awburst,
    output wire [1023:0] m_wdata,
    input wire [7:0] m_bid,
    input wire [3:0] m_bresp
);
    wire [1:0] head_valid,head_good,reserve_can,reserve_take,engine_can;
    wire [463:0] head_desc;
    wire command_valid,command_ready,command_channel;
    wire [231:0] command_desc;
    wire dispatch_fetch_valid,dispatch_fetch_ready;
    wire done_ready,dispatch_fault,engine_fault,engine_drained,local_empty;
    wire engine_cmd_ready,engine_data_ready;
    reg adapter_fault,lease_valid,command_seen,payload_active,done_pending;
    reg [231:0] lease_desc;
    reg lease_channel;
    reg [8:0] received;
    wire [1:0] descriptor_good;
    wire selected_channel=reserve_take[1];
    wire [231:0] selected_desc=head_desc[selected_channel*232+:232];
    wire engine_cmd_valid=|reserve_take;
    wire engine_data_valid=payload_active && payload_valid;
    wire payload_fire=engine_data_valid && engine_data_ready;
    wire [8:0] expected_beats={1'b0,lease_desc[7:0]}+1;
    wire bad_command_echo=command_valid && (!lease_valid ||
        command_channel!=lease_channel || command_desc!=lease_desc);

    assign fault=adapter_fault || engine_fault || dispatch_fault || fault_in;
    assign command_ready=lease_valid && !command_seen && !bad_command_echo;
    assign fetch_valid=dispatch_fetch_valid && lease_valid && command_seen && !payload_active && !done_pending;
    assign dispatch_fetch_ready=fetch_ready && lease_valid && command_seen && !payload_active && !done_pending;
    assign payload_ready=payload_active && engine_data_ready;
    assign descriptor_good=head_good;
    assign reserve_can=engine_can & head_good & {2{!lease_valid}};


    ntrpu_destination_registered dispatch (
        .clk(clk),.rst_n(rst_n),.stop(stop),.fault_in(fault_in || adapter_fault || engine_fault),
        .submit_valid(submit_valid),.submit_ready(submit_ready),.submit_desc(submit_desc),
        .head_valid(head_valid),.head_good(head_good),.head_desc(head_desc),.queued(queued),
        .reserve_can(reserve_can),.reserve_take(reserve_take),
        .cmd_valid(command_valid),.cmd_ready(command_ready),.cmd_channel(command_channel),.cmd_desc(command_desc),
        .fetch_valid(dispatch_fetch_valid),.fetch_ready(dispatch_fetch_ready),.fetch_channel(fetch_channel),.fetch_desc(fetch_desc),
        .frame_done_valid(done_pending),.frame_done_ready(done_ready),
        .frame_done_tag(lease_desc[199:136]),.frame_done_epoch(lease_desc[231:200]),.frame_done_beats(received),
        .fence_valid(fence_valid),.fence_ready(fence_ready),.fence_id(fence_id),
        .fence_done_valid(fence_done_valid),.fence_done_ready(fence_done_ready),.fence_done_id(fence_done_id),
        .sink_drained(engine_drained && !lease_valid && !payload_active && !done_pending),
        .fault(dispatch_fault),.local_empty(local_empty),.drained(drained)
    );

    ntrpu_peer_multibuffer_reserved receiver (
        .clk(clk),.rst_n(rst_n),.stop(fault_in || adapter_fault || dispatch_fault),
        .reserve_can(engine_can),.cmd_valid(engine_cmd_valid),.cmd_ready(engine_cmd_ready),
        .cmd_channel(selected_channel),.cmd_addr(selected_desc[135:72]),
        .cmd_tag(selected_desc[199:136]),.cmd_len(selected_desc[7:0]),
        .data_valid(engine_data_valid),.data_ready(engine_data_ready),.data(payload_data),
        .fault(engine_fault),.drained(engine_drained),.*
    );

    always @(posedge clk)begin
        if(!rst_n)begin
            adapter_fault<=0;lease_valid<=0;command_seen<=0;payload_active<=0;done_pending<=0;
            lease_desc<=0;lease_channel<=0;received<=0;
        end else begin
            if(bad_command_echo || (|(head_valid & ~descriptor_good)))adapter_fault<=1;
            if(engine_cmd_valid)begin
                if(!engine_cmd_ready || lease_valid || reserve_take==3)adapter_fault<=1;
                else begin
                    lease_valid<=1;lease_desc<=selected_desc;lease_channel<=selected_channel;
                    command_seen<=0;received<=0;
                end
            end
            if(command_valid && command_ready)command_seen<=1;
            if(fetch_valid && fetch_ready)payload_active<=1;
            if(payload_fire)begin
                received<=received+1'b1;
                if(received+1'b1==expected_beats)begin payload_active<=0;done_pending<=1;end
            end
            if(done_pending && done_ready)begin
                done_pending<=0;lease_valid<=0;command_seen<=0;
            end
        end
    end
endmodule
