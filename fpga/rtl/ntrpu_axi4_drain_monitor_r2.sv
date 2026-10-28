`timescale 1ns / 1ps
// Passive, configuration-retained AXI4 observer for the 512-bit NTRPU fabric.
// No operational reset input; source_fenced MUST cover upstream queues/CDC.
// This is neither an admission fence nor a response containment boundary.
// WRITE_DEPTH bounds AW and completed W obligations UNTIL B retires them.
// READ_DEPTH is PER ID. Overflow makes inventory permanently untrustworthy.
module ntrpu_axi4_drain_monitor_r2 #(
    parameter integer ID_WIDTH=4, WRITE_DEPTH=8, READ_DEPTH=8
) (
    input wire clk, cold_rst_n,
    input wire awvalid, awready, wvalid, wready, bvalid, bready,
    input wire arvalid, arready, rvalid, rready,
    input wire [63:0] awaddr, araddr,
    input wire [ID_WIDTH-1:0] awid, bid, arid, rid,
    input wire [7:0] awlen, arlen,
    input wire [2:0] awsize, arsize, awprot, arprot,
    input wire [1:0] awburst, arburst, bresp, rresp,
    input wire [3:0] awcache, arcache,
    input wire awlock, arlock, wlast, rlast,
    input wire [511:0] wdata, rdata,
    input wire [63:0] wstrb,
    input wire req_valid, source_fenced,
    input wire [63:0] req_tag,
    output wire ack_valid,
    output wire [63:0] ack_tag,
    // capacity, orphan B, orphan R, W boundary, R boundary, stalled payload
    output reg [5:0] fault_sources=0,
    output wire protocol_fault,
    output reg [$clog2(WRITE_DEPTH+1)-1:0] aw_pending=0, w_pending=0,
    output wire [$clog2((1<<ID_WIDTH)*READ_DEPTH+1)-1:0] ar_pending,
    output reg [8:0] w_partial=0
);
    localparam integer IDS=1<<ID_WIDTH;
    localparam integer WC=$clog2(WRITE_DEPTH+1), RC=$clog2(READ_DEPTH+1);
    localparam integer WP=(WRITE_DEPTH>1)?$clog2(WRITE_DEPTH):1;
    localparam integer RP=(READ_DEPTH>1)?$clog2(READ_DEPTH):1;
    localparam integer RT=$clog2(IDS*READ_DEPTH+1), AP=85+ID_WIDTH;
    wire af=awvalid && awready, wf=wvalid && wready;
    wire bf=bvalid && bready, rf=rvalid && rready, arf=arvalid && arready;
    wire any_valid=awvalid || wvalid || bvalid || arvalid || rvalid;

    reg [ID_WIDTH-1:0] aw_ids[0:WRITE_DEPTH-1];
    reg [8:0] aw_beats[0:WRITE_DEPTH-1], w_beats[0:WRITE_DEPTH-1];
    reg [WP-1:0] aw_head=0, aw_tail=0, w_head=0, w_tail=0;
    reg [WC-1:0] aw_queued=0, w_queued=0;
    reg [WC-1:0] b_credit[0:IDS-1];
    reg [RC-1:0] reads[0:IDS-1];
    reg [RP-1:0] read_head[0:IDS-1], read_tail[0:IDS-1];
    reg [8:0] read_beats[0:IDS-1][0:READ_DEPTH-1];
    reg [RT-1:0] total_reads;
    integer k;
    always @* begin
        total_reads=0;
        for(integer n=0;n<IDS;n=n+1) total_reads=total_reads+reads[n];
    end
    assign ar_pending=total_reads;
    assign protocol_fault=|fault_sources;

    // Pair only histories from PRIOR accepted edges. A legal B on the first
    // edge after AW/WLAST can consume this combinational pairing directly.
    wire pair_ready=aw_queued!=0 && w_queued!=0;
    wire pair_good=pair_ready && aw_beats[aw_head]==w_beats[w_head];
    wire [ID_WIDTH-1:0] pair_id=aw_ids[aw_head];
    wire b_available=b_credit[bid]!=0 || (pair_good && pair_id==bid);
    wire r_available=reads[rid]!=0;
    wire r_final=r_available && read_beats[rid][read_head[rid]]==1;
    wire br=bf && b_available, rr=rf && r_available && r_final && rlast;
    wire wc=wf && wlast;

    // The partial W burst follows all already completed, unpaired W bursts.
    // If its AW exists, reaching LEN+1 without LAST is already a violation.
    // Continuous evaluation includes time zero; declaration-initialized heads
    // need not change before the first AW overlaps an unfinished W burst.
    // One subtraction suffices: head < DEPTH and queued <= DEPTH.
    wire [WC:0] partial_sum={1'b0,aw_head}+{1'b0,w_queued};
    wire [WP-1:0] partial_index=(partial_sum>=WRITE_DEPTH)?
        partial_sum-WRITE_DEPTH:partial_sum;
    wire partial_bad=aw_queued>w_queued && w_partial!=0 &&
                     w_partial>=aw_beats[partial_index];

    wire [AP-1:0] aw_payload={awaddr,awid,awlen,awsize,awprot,awburst,awcache,awlock};
    wire [AP-1:0] ar_payload={araddr,arid,arlen,arsize,arprot,arburst,arcache,arlock};
    wire [576:0] w_payload={wdata,wstrb,wlast};
    wire [ID_WIDTH+1:0] b_payload={bid,bresp};
    wire [ID_WIDTH+514:0] r_payload={rdata,rid,rresp,rlast};
    reg [AP-1:0] held_aw=0, held_ar=0;
    reg [576:0] held_w=0;
    reg [ID_WIDTH+1:0] held_b=0;
    reg [ID_WIDTH+514:0] held_r=0;
    reg [4:0] stalled=0;
    wire stall_bad=(stalled[0] && (!awvalid || aw_payload!=held_aw)) ||
                   (stalled[1] && (!wvalid || w_payload!=held_w)) ||
                   (stalled[2] && (!bvalid || b_payload!=held_b)) ||
                   (stalled[3] && (!arvalid || ar_payload!=held_ar)) ||
                   (stalled[4] && (!rvalid || r_payload!=held_r));
    wire capacity_bad=(af && aw_pending==WRITE_DEPTH && !br) ||
                      (wf && w_partial==0 && w_pending==WRITE_DEPTH && !br) ||
                      (arf && reads[arid]==READ_DEPTH && !(rr && rid==arid));
    wire [5:0] new_fault={stall_bad,
        rvalid && r_available && (rlast!=r_final),
        (pair_ready && !pair_good) || partial_bad ||
            (wf && (w_partial==256 || (w_partial==255 && !wlast))),
        rvalid && !r_available, bvalid && !b_available, capacity_bad};

    function [WP-1:0] wnext(input [WP-1:0] ptr);
        wnext=(ptr==WRITE_DEPTH-1)?0:ptr+1'b1;
    endfunction
    function [RP-1:0] rnext(input [RP-1:0] ptr);
        rnext=(ptr==READ_DEPTH-1)?0:ptr+1'b1;
    endfunction
    reg captured=0;
    reg [63:0] captured_tag=0;
    reg [1:0] quiet_count=0;
    wire empty=aw_pending==0 && w_pending==0 && total_reads==0 &&
               aw_queued==0 && w_queued==0 && w_partial==0;
    assign ack_tag=captured_tag;
    assign ack_valid=cold_rst_n && captured && req_valid && captured_tag==req_tag &&
        source_fenced && !protocol_fault && empty && !any_valid && quiet_count==2;

    initial begin
        for(integer i=0;i<IDS;i=i+1) begin
            b_credit[i]=0;reads[i]=0;read_head[i]=0;read_tail[i]=0;
        end
    end
    always @(posedge clk) begin
        if(!cold_rst_n) begin
            fault_sources<=0;aw_pending<=0;w_pending<=0;w_partial<=0;
            aw_head<=0;aw_tail<=0;w_head<=0;w_tail<=0;aw_queued<=0;w_queued<=0;
            for(k=0;k<IDS;k=k+1) begin
                b_credit[k]<=0;reads[k]<=0;read_head[k]<=0;read_tail[k]<=0;
            end
            stalled<=0;captured<=0;captured_tag<=0;quiet_count<=0;
        end else begin
            fault_sources<=fault_sources | new_fault;
            stalled<={rvalid&&!rready,arvalid&&!arready,bvalid&&!bready,wvalid&&!wready,awvalid&&!awready};
            held_aw<=aw_payload;held_w<=w_payload;held_b<=b_payload;
            held_ar<=ar_payload;held_r<=r_payload;
            // Freeze inventories on corruption, including the detecting edge.
            // All fault sources remain independent sticky registered outputs.
            if(!protocol_fault && !(|new_fault)) begin
                case({af,br})
                    2'b10:aw_pending<=aw_pending+1'b1;
                    2'b01:aw_pending<=aw_pending-1'b1;
                endcase
                case({wc,br})
                    2'b10:w_pending<=w_pending+1'b1;
                    2'b01:w_pending<=w_pending-1'b1;
                endcase
                case({af,pair_good})
                    2'b10:aw_queued<=aw_queued+1'b1;
                    2'b01:aw_queued<=aw_queued-1'b1;
                endcase
                case({wc,pair_good})
                    2'b10:w_queued<=w_queued+1'b1;
                    2'b01:w_queued<=w_queued-1'b1;
                endcase
                if(af) begin
                    aw_ids[aw_tail]<=awid;aw_beats[aw_tail]<={1'b0,awlen}+9'd1;
                    aw_tail<=wnext(aw_tail);
                end
                if(wf) begin
                    if(wlast) begin
                        w_beats[w_tail]<=w_partial+9'd1;
                        w_tail<=wnext(w_tail);w_partial<=0;
                    end else w_partial<=w_partial+9'd1;
                end
                if(pair_good) begin aw_head<=wnext(aw_head);w_head<=wnext(w_head);end
                for(k=0;k<IDS;k=k+1) begin
                    case({pair_good && pair_id==k,br && bid==k})
                        2'b10:b_credit[k]<=b_credit[k]+1'b1;
                        2'b01:b_credit[k]<=b_credit[k]-1'b1;
                    endcase
                    case({arf && arid==k,rr && rid==k})
                        2'b10:reads[k]<=reads[k]+1'b1;
                        2'b01:reads[k]<=reads[k]-1'b1;
                    endcase
                end
                if(arf) begin
                    read_beats[arid][read_tail[arid]]<={1'b0,arlen}+9'd1;
                    read_tail[arid]<=rnext(read_tail[arid]);
                end
                if(rf && r_available) begin
                    if(rr) read_head[rid]<=rnext(read_head[rid]);
                    else read_beats[rid][read_head[rid]]<=read_beats[rid][read_head[rid]]-9'd1;
                end
            end
            if(!req_valid) begin captured<=0;quiet_count<=0;end
            else if(!captured || captured_tag!=req_tag) begin
                captured<=1;captured_tag<=req_tag;quiet_count<=0;
            end else if(!source_fenced || protocol_fault || !empty || any_valid) quiet_count<=0;
            else if(quiet_count<2) quiet_count<=quiet_count+1'b1;
        end
    end
    // synthesis translate_off
    initial if(ID_WIDTH<1 || ID_WIDTH>4 || WRITE_DEPTH<2 || READ_DEPTH<2)
        $fatal(1,"AXI4_DRAIN_PARAMETERS");
    // synthesis translate_on
endmodule
