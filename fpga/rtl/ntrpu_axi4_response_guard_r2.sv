`timescale 1ns / 1ps
// Causal AXI4 response boundary. Observe requests at the protected interface;
// c_* is the untrusted return, h_* the protected consumer. Configuration reset
// only. A quarantined channel never reopens when later requests arrive.
// One output slot/channel preserves a credited response across later faults.
// The guard proves ID/burst credit, not the semantic correctness of data/RESP.
module ntrpu_axi4_response_guard_r2 #(
    parameter integer ID_WIDTH=4, WRITE_DEPTH=32, READ_DEPTH=8
) (
    input wire clk,cold_rst_n,
    input wire awvalid,awready,wvalid,wready,wlast,arvalid,arready,
    input wire [ID_WIDTH-1:0] awid,arid,
    input wire [7:0] awlen,arlen,
    input wire c_bvalid,c_rvalid,c_rlast,
    output wire c_bready,c_rready,
    input wire [ID_WIDTH-1:0] c_bid,c_rid,
    input wire [1:0] c_bresp,c_rresp,
    input wire [511:0] c_rdata,
    output wire h_bvalid,h_rvalid,h_rlast,
    input wire h_bready,h_rready,
    output wire [ID_WIDTH-1:0] h_bid,h_rid,
    output wire [1:0] h_bresp,h_rresp,
    output wire [511:0] h_rdata,
    // Independent registered sources: B quarantine, R quarantine, W/R inventory.
    output wire [3:0] fault_sources,
    output wire aw_capacity,w_capacity,ar_capacity,local_idle,
    output reg [$clog2(WRITE_DEPTH+1)-1:0] aw_pending=0,w_pending=0,
    output wire [$clog2((1<<ID_WIDTH)*READ_DEPTH+1)-1:0] ar_pending,
    output reg [8:0] w_partial=0
);
    localparam integer IDS=1<<ID_WIDTH,WC=$clog2(WRITE_DEPTH+1);
    localparam integer RC=$clog2(READ_DEPTH+1),WP=$clog2(WRITE_DEPTH);
    localparam integer RP=$clog2(READ_DEPTH),RT=$clog2(IDS*READ_DEPTH+1);
    reg quarantine_b=0,quarantine_r=0,write_fault=0,read_fault=0;
    assign fault_sources={read_fault,write_fault,quarantine_r,quarantine_b};
    reg b_hold=0,r_hold=0;
    reg [ID_WIDTH-1:0] b_id=0,r_id=0;
    reg [1:0] b_resp=0,r_resp=0;
    reg [511:0] r_data=0;
    reg r_last=0;
    assign h_bvalid=cold_rst_n && b_hold;
    assign h_rvalid=cold_rst_n && r_hold;
    assign h_bid=b_id;assign h_bresp=b_resp;
    assign h_rid=r_id;assign h_rresp=r_resp;assign h_rdata=r_data;assign h_rlast=r_last;
    wire hb=h_bvalid && h_bready,hr=h_rvalid && h_rready;
    wire af=awvalid && awready,wf=wvalid && wready,arf=arvalid && arready;
    wire wc=wf && wlast;
    reg [ID_WIDTH-1:0] aw_ids[0:WRITE_DEPTH-1];
    reg [8:0] aw_beats[0:WRITE_DEPTH-1],w_beats[0:WRITE_DEPTH-1];
    reg [WP-1:0] aw_head=0,aw_tail=0,w_head=0,w_tail=0;
    reg [WC-1:0] aw_queued=0,w_queued=0,b_credit[0:IDS-1];
    reg [RC-1:0] reads[0:IDS-1],read_pending[0:IDS-1];
    reg [RP-1:0] read_head[0:IDS-1],read_tail[0:IDS-1];
    reg [8:0] read_beats[0:IDS-1][0:READ_DEPTH-1];
    reg [RT-1:0] total_reads;
    always @* begin
        total_reads=0;
        for(integer i=0;i<IDS;i=i+1)total_reads=total_reads+read_pending[i];
    end
    assign ar_pending=total_reads;
    // Capacity includes responses held above the guard until h_* consumes them.
    // Conservative: a retiring response makes room on the following cycle.
    assign aw_capacity=cold_rst_n && !write_fault && aw_pending<WRITE_DEPTH;
    assign w_capacity=cold_rst_n && !write_fault && (w_partial!=0 || w_pending<WRITE_DEPTH);
    assign ar_capacity=cold_rst_n && !read_fault && read_pending[arid]<READ_DEPTH;
    assign local_idle=cold_rst_n && !b_hold && !r_hold && aw_pending==0 &&
        w_pending==0 && w_partial==0 && total_reads==0 && !(|fault_sources) &&
        !(c_bvalid || c_rvalid);
    wire pair_ready=aw_queued!=0 && w_queued!=0;
    wire pair_good=pair_ready && aw_beats[aw_head]==w_beats[w_head];
    wire [ID_WIDTH-1:0] pair_id=aw_ids[aw_head];
    wire b_available=b_credit[c_bid]!=0 || (pair_good && pair_id==c_bid);
    wire r_available=reads[c_rid]!=0;
    wire r_final=r_available && read_beats[c_rid][read_head[c_rid]]==1;
    // Continuous evaluation includes time zero; declaration-initialized heads
    // need not change before the first AW overlaps an unfinished W burst.
    // One subtraction suffices: head < DEPTH and queued <= DEPTH.
    wire [WC:0] partial_sum={1'b0,aw_head}+{1'b0,w_queued};
    wire [WP-1:0] partial_index=(partial_sum>=WRITE_DEPTH)?
        partial_sum-WRITE_DEPTH:partial_sum;
    wire write_bad=(af && aw_pending==WRITE_DEPTH && !hb) ||
        (wf && w_partial==0 && w_pending==WRITE_DEPTH && !hb) ||
        (pair_ready && !pair_good) ||
        (aw_queued>w_queued && w_partial!=0 && w_partial>=aw_beats[partial_index]) ||
        (wf && (w_partial==256 || (w_partial==255 && !wlast)));
    wire read_bad=arf && read_pending[arid]==READ_DEPTH && !(hr && h_rlast && h_rid==arid);
    // Check invalid offers, including when READY is low. A later credit cannot
    // retroactively authorize an orphan already observed here.
    wire [ID_WIDTH+1:0] b_payload={c_bid,c_bresp};
    wire [ID_WIDTH+514:0] r_payload={c_rdata,c_rid,c_rresp,c_rlast};
    reg b_stalled=0,r_stalled=0;
    reg [ID_WIDTH+1:0] held_b=0;
    reg [ID_WIDTH+514:0] held_r=0;
    wire b_changed=b_stalled && (!c_bvalid || b_payload!=held_b);
    wire r_changed=r_stalled && (!c_rvalid || r_payload!=held_r);
    wire b_bad=(c_bvalid && !b_available) || b_changed;
    wire r_bad=(c_rvalid && (!r_available || c_rlast!=r_final)) || r_changed;
    assign c_bready=cold_rst_n && (!b_hold || h_bready) && b_available &&
        !quarantine_b && !write_fault && !write_bad && !b_changed;
    assign c_rready=cold_rst_n && (!r_hold || h_rready) && r_available && c_rlast==r_final &&
        !quarantine_r && !read_fault && !read_bad && !r_changed;
    wire cb=c_bvalid && c_bready,cr=c_rvalid && c_rready;
    wire cr_last=cr && r_final,hr_last=hr && h_rlast;
    function [WP-1:0] wnext(input [WP-1:0] p);
        wnext=(p==WRITE_DEPTH-1)?0:p+1'b1;
    endfunction
    function [RP-1:0] rnext(input [RP-1:0] p);
        rnext=(p==READ_DEPTH-1)?0:p+1'b1;
    endfunction
    initial begin
        for(integer i=0;i<IDS;i=i+1)begin
            b_credit[i]=0;reads[i]=0;read_pending[i]=0;read_head[i]=0;read_tail[i]=0;
        end
    end
    integer k;
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            quarantine_b<=0;quarantine_r<=0;write_fault<=0;read_fault<=0;
            b_hold<=0;r_hold<=0;b_id<=0;r_id<=0;b_resp<=0;r_resp<=0;r_data<=0;r_last<=0;
            aw_pending<=0;w_pending<=0;w_partial<=0;
            aw_head<=0;aw_tail<=0;w_head<=0;w_tail<=0;aw_queued<=0;w_queued<=0;
            b_stalled<=0;r_stalled<=0;held_b<=0;held_r<=0;
            for(k=0;k<IDS;k=k+1)begin
                b_credit[k]<=0;reads[k]<=0;read_pending[k]<=0;read_head[k]<=0;read_tail[k]<=0;
            end
        end else begin
            quarantine_b<=quarantine_b || b_bad;
            quarantine_r<=quarantine_r || r_bad;
            write_fault<=write_fault || write_bad;read_fault<=read_fault || read_bad;
            b_stalled<=c_bvalid && !c_bready;r_stalled<=c_rvalid && !c_rready;
            held_b<=b_payload;held_r<=r_payload;
            // Never gate a held protected offer with any newly observed fault.
            if(hb)b_hold<=0;
            if(hr)r_hold<=0;
            if(cb)begin b_hold<=1;b_id<=c_bid;b_resp<=c_bresp;end
            if(cr)begin r_hold<=1;r_id<=c_rid;r_resp<=c_rresp;r_data<=c_rdata;r_last<=c_rlast;end
            if(!write_fault && !write_bad)begin
                case({af,hb})2'b10:aw_pending<=aw_pending+1'b1;2'b01:aw_pending<=aw_pending-1'b1;endcase
                case({wc,hb})2'b10:w_pending<=w_pending+1'b1;2'b01:w_pending<=w_pending-1'b1;endcase
                case({af,pair_good})2'b10:aw_queued<=aw_queued+1'b1;2'b01:aw_queued<=aw_queued-1'b1;endcase
                case({wc,pair_good})2'b10:w_queued<=w_queued+1'b1;2'b01:w_queued<=w_queued-1'b1;endcase
                if(af)begin aw_ids[aw_tail]<=awid;aw_beats[aw_tail]<={1'b0,awlen}+9'd1;aw_tail<=wnext(aw_tail);end
                if(wf)begin
                    if(wlast)begin w_beats[w_tail]<=w_partial+9'd1;w_tail<=wnext(w_tail);w_partial<=0;end
                    else w_partial<=w_partial+9'd1;
                end
                if(pair_good)begin aw_head<=wnext(aw_head);w_head<=wnext(w_head);end
                for(k=0;k<IDS;k=k+1)begin
                    case({pair_good && pair_id==k,cb && c_bid==k})
                        2'b10:b_credit[k]<=b_credit[k]+1'b1;
                        2'b01:b_credit[k]<=b_credit[k]-1'b1;
                    endcase
                end
            end
            if(!read_fault && !read_bad)begin
                for(k=0;k<IDS;k=k+1)begin
                    case({arf && arid==k,cr_last && c_rid==k})
                        2'b10:reads[k]<=reads[k]+1'b1;2'b01:reads[k]<=reads[k]-1'b1;
                    endcase
                    case({arf && arid==k,hr_last && h_rid==k})
                        2'b10:read_pending[k]<=read_pending[k]+1'b1;2'b01:read_pending[k]<=read_pending[k]-1'b1;
                    endcase
                end
                if(arf)begin read_beats[arid][read_tail[arid]]<={1'b0,arlen}+9'd1;read_tail[arid]<=rnext(read_tail[arid]);end
                if(cr)begin
                    if(cr_last)read_head[c_rid]<=rnext(read_head[c_rid]);
                    else read_beats[c_rid][read_head[c_rid]]<=read_beats[c_rid][read_head[c_rid]]-9'd1;
                end
            end
        end
    end
    // synthesis translate_off
    initial if(ID_WIDTH<1 || ID_WIDTH>4 || WRITE_DEPTH<2 || READ_DEPTH<2)
        $fatal(1,"AXI4_RESPONSE_GUARD_PARAMETERS");
    // synthesis translate_on
endmodule
