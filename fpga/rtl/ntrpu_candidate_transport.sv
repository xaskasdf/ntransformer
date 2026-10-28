`timescale 1ns/1ps
// Same-clock transport boundary for the isolated candidate core, not a PCIe IP
// or CDC bridge. Descriptors are already authorized; reset requires a dead epoch.
// A stop is latched until reset. Accepted ingress drains before core stop.
// Fault quarantines unreserved descriptors; committed fetch/payload/CPL drain.
// Memory AW/W/B already have two-entry elastic queues inside the candidate.
module ntrpu_candidate_transport (
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
    localparam [2:0] OPEN=0, WAIT_INGRESS=1, SEND_FENCE=2, WAIT_CORE=3, REPORT=4;
    reg [2:0] fence_state;
    reg [63:0] saved_fence;
    reg stop_seen,transport_fault;
    reg [8:0] payload_budget;
    reg [1:0] ingress_count [0:1];
    wire [1:0] ingress_ready,ingress_empty,completion_empty;
    wire [1:0] i_submit_valid,i_submit_ready,i_cpl_valid,i_cpl_ready;
    wire [463:0] i_submit_desc;
    wire [127:0] i_cpl_tag;
    wire [3:0] i_cpl_status;
    wire [7:0] core_queued;
    wire core_fault,core_drained,fetch_empty,payload_empty;
    wire i_fetch_valid,i_fetch_ready,i_fetch_channel;
    wire [231:0] i_fetch_desc;
    wire i_payload_valid,i_payload_ready,payload_stage_ready;
    wire [511:0] i_payload_data;
    wire i_fence_ready,i_fence_done_valid,i_fence_done_ready;
    wire [63:0] i_fence_done_id;
    wire stopped=stop_seen || stop;
    wire ingress_clear=&ingress_empty;
    wire completion_clear=&completion_empty;
    wire i_stop=stopped && ingress_clear;
    wire admit=rst_n && !stopped && !fault && fence_state==OPEN && !fence_valid;
    wire fetch_fire=fetch_valid && fetch_ready;
    wire payload_fire=payload_valid && payload_ready;
    wire fence_fire=fence_valid && fence_ready;
    wire i_fence_valid=rst_n && fence_state==SEND_FENCE;

    assign fault=core_fault || transport_fault || fault_in;
    assign submit_ready=ingress_ready & {2{admit}};
    assign payload_ready=payload_stage_ready && payload_budget!=0;
    assign fence_ready=rst_n && !fault && fence_state==OPEN;
    // Success cannot overtake externally buffered completions. Once REPORT is
    // offered, VALID/id persist even if a later stop/fault arrives.
    assign i_fence_done_ready=rst_n && fence_state==WAIT_CORE && !fault &&
        completion_clear && fetch_empty && payload_empty && payload_budget==0;
    assign fence_done_valid=rst_n && fence_state==REPORT;
    assign fence_done_id=saved_fence;
    assign drained=core_drained && ingress_clear && completion_clear &&
        fetch_empty && payload_empty && payload_budget==0 && fence_state==OPEN &&
        !fence_valid && !(|(submit_valid & submit_ready));

    genvar g;
    generate for(g=0;g<2;g=g+1)begin: boundary
        wire push=submit_valid[g] && submit_ready[g];
        wire pop=i_submit_valid[g] && i_submit_ready[g];
        ntrpu_rv_elastic #(.WIDTH(232)) ingress (
            .clk(clk),.rst_n(rst_n),.s_valid(submit_valid[g] && admit),
            .s_ready(ingress_ready[g]),.s_data(submit_desc[g*232+:232]),
            .m_valid(i_submit_valid[g]),.m_ready(i_submit_ready[g]),
            .m_data(i_submit_desc[g*232+:232]),.empty(ingress_empty[g])
        );
        ntrpu_rv_elastic #(.WIDTH(66)) completion (
            .clk(clk),.rst_n(rst_n),.s_valid(i_cpl_valid[g]),.s_ready(i_cpl_ready[g]),
            .s_data({i_cpl_tag[g*64+:64],i_cpl_status[g*2+:2]}),
            .m_valid(cpl_valid[g]),.m_ready(cpl_ready[g]),
            .m_data({cpl_tag[g*64+:64],cpl_status[g*2+:2]}),.empty(completion_empty[g])
        );
        // Public queued counts include both accepted ingress slots and core FIFO.
        assign queued[g*4+:4]=core_queued[g*4+:4]+{2'b0,ingress_count[g]};
        always @(posedge clk)begin
            if(!rst_n)ingress_count[g]<=0;
            else case({push,pop})
                2'b10: ingress_count[g]<=ingress_count[g]+1'b1;
                2'b01: ingress_count[g]<=ingress_count[g]-1'b1;
                default: begin end
            endcase
        end
    end endgenerate
    ntrpu_rv_elastic #(.WIDTH(233)) fetch (
        .clk(clk),.rst_n(rst_n),.s_valid(i_fetch_valid),.s_ready(i_fetch_ready),
        .s_data({i_fetch_channel,i_fetch_desc}),.m_valid(fetch_valid),.m_ready(fetch_ready),
        .m_data({fetch_channel,fetch_desc}),.empty(fetch_empty)
    );
    ntrpu_rv_elastic #(.WIDTH(512)) payload (
        .clk(clk),.rst_n(rst_n),.s_valid(payload_valid && payload_budget!=0),
        .s_ready(payload_stage_ready),.s_data(payload_data),.m_valid(i_payload_valid),
        .m_ready(i_payload_ready),.m_data(i_payload_data),.empty(payload_empty)
    );
    ntrpu_candidate_dispatch core (
        .stop(i_stop),.fault_in(fault_in || transport_fault),
        .submit_valid(i_submit_valid),.submit_ready(i_submit_ready),.submit_desc(i_submit_desc),
        .fetch_valid(i_fetch_valid),.fetch_ready(i_fetch_ready),.fetch_channel(i_fetch_channel),.fetch_desc(i_fetch_desc),
        .payload_valid(i_payload_valid),.payload_ready(i_payload_ready),.payload_data(i_payload_data),
        .fence_valid(i_fence_valid),.fence_ready(i_fence_ready),.fence_id(saved_fence),
        .fence_done_valid(i_fence_done_valid),.fence_done_ready(i_fence_done_ready),.fence_done_id(i_fence_done_id),
        .cpl_valid(i_cpl_valid),.cpl_ready(i_cpl_ready),.cpl_tag(i_cpl_tag),.cpl_status(i_cpl_status),
        .fault(core_fault),.drained(core_drained),.queued(core_queued),.*
    );
    always @(posedge clk)begin
        if(!rst_n)begin
            stop_seen<=0;transport_fault<=0;payload_budget<=0;
            fence_state<=OPEN;saved_fence<=0;
        end else begin
            if(stop)stop_seen<=1;
            if(fault_in || core_fault)transport_fault<=1;
            if(fetch_fire)begin
                if(payload_budget!=0)transport_fault<=1;
                else payload_budget<={1'b0,fetch_desc[7:0]}+9'd1;
            end else if(payload_fire)payload_budget<=payload_budget-1'b1;
            case(fence_state)
                OPEN: if(fence_fire)begin saved_fence<=fence_id;fence_state<=WAIT_INGRESS;end
                WAIT_INGRESS: if(ingress_clear)fence_state<=SEND_FENCE;
                SEND_FENCE: if(i_fence_ready)fence_state<=WAIT_CORE;
                WAIT_CORE: if(i_fence_done_valid && i_fence_done_ready)begin
                    if(i_fence_done_id!=saved_fence)transport_fault<=1;
                    else fence_state<=REPORT;
                end
                REPORT: if(fence_done_ready)fence_state<=OPEN;
                default: transport_fault<=1;
            endcase
        end
    end
endmodule
