`timescale 1ns/1ps
// Experimental write-only guard. One ordered downstream ID (zero).
// DEPTH credits include local rejects and replies not yet consumed upstream.
// Input AW/W must be AXI compliant; cold/coordinated reset is not an AXI abort.
module ntrpu_axi_write_capacity_queue #(
    parameter integer MEMORY_ADDR_WIDTH=37,
    parameter integer DEPTH=4,
    parameter integer PTR_BITS=DEPTH<=1 ? 1:$clog2(DEPTH),
    parameter integer COUNT_BITS=$clog2(DEPTH+1)
) (
    input wire clk, rst_n, admit,
    output wire idle,
    input wire [63:0] s_axi_awaddr,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize, s_axi_awprot,
    input wire [1:0] s_axi_awburst,
    input wire [3:0] s_axi_awid, s_axi_awcache,
    input wire s_axi_awlock, s_axi_awvalid,
    output wire s_axi_awready,
    input wire [511:0] s_axi_wdata,
    input wire [63:0] s_axi_wstrb,
    input wire s_axi_wlast, s_axi_wvalid,
    output wire s_axi_wready,
    output reg [3:0] s_axi_bid,
    output reg [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    output wire [MEMORY_ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize, m_axi_awprot,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awid, m_axi_awcache,
    output wire m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [3:0] m_axi_bid,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    localparam [64:0] CAPACITY=65'd1<<MEMORY_ADDR_WIDTH;
    reg [PTR_BITS-1:0] head, tail, aw_head, w_head;
    reg [COUNT_BITS-1:0] used, aw_count, w_count;
    reg [DEPTH-1:0] allowed, aw_done, w_done;
    reg [MEMORY_ADDR_WIDTH-1:0] address[0:DEPTH-1];
    reg [7:0] length[0:DEPTH-1];
    reg [8:0] remaining[0:DEPTH-1];
    reg [3:0] identity[0:DEPTH-1], cache[0:DEPTH-1];
    reg [2:0] protection[0:DEPTH-1];
    reg reply_valid;

    function automatic [PTR_BITS-1:0] next_ptr(input [PTR_BITS-1:0] pointer);
        return pointer==DEPTH-1 ? PTR_BITS'(0):pointer+1'b1;
    endfunction
    function automatic permitted(input [63:0] addr, input [7:0] len,
        input [2:0] size, input [1:0] burst, input locked, input [3:0] id);
        reg [8:0] beats;
        reg [64:0] end_address;
        reg [14:0] page_end;
        begin
            beats={1'b0,len}+9'd1;
            end_address={1'b0,addr}+{50'b0,beats,6'b0};
            page_end={3'b0,addr[11:0]}+{beats,6'b0};
            return size==6 && burst==1 && !locked && id==0 && addr[5:0]==0 &&
                   page_end<=4096 && {1'b0,addr}<CAPACITY && end_address<=CAPACITY;
        end
    endfunction

    wire push=s_axi_awvalid && s_axi_awready;
    wire retire=s_axi_bvalid && s_axi_bready;
    wire aw_step=rst_n && aw_count!=0 && (!allowed[aw_head] || m_axi_awready);
    wire input_w=s_axi_wvalid && s_axi_wready;
    wire input_last=input_w && remaining[w_head]==1;
    wire fifo_ready, fifo_empty;
    wire [PTR_BITS-1:0] output_owner;

    assign s_axi_awready=rst_n && admit && used<DEPTH;
    assign s_axi_wready=rst_n && w_count!=0 && (!allowed[w_head] || fifo_ready);
    assign m_axi_awvalid=rst_n && aw_count!=0 && allowed[aw_head];
    assign m_axi_awaddr=address[aw_head];
    assign m_axi_awlen=length[aw_head];
    assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;
    assign m_axi_awid=4'b0;
    assign m_axi_awcache=cache[aw_head];
    assign m_axi_awprot=protection[aw_head];
    assign m_axi_awlock=1'b0;
    assign m_axi_bready=rst_n && used!=0 && allowed[head] && aw_done[head] &&
                        w_done[head] && !reply_valid;
    assign s_axi_bvalid=rst_n && reply_valid;
    assign idle=rst_n && used==0 && fifo_empty && !reply_valid;

    // Store ownership until the last external W handshake, not only input WLAST.
    ntrpu_rv_elastic #(.WIDTH(577+PTR_BITS)) writes (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_axi_wvalid && w_count!=0 && allowed[w_head]), .s_ready(fifo_ready),
        .s_data({w_head,s_axi_wlast,s_axi_wstrb,s_axi_wdata}),
        .m_valid(m_axi_wvalid), .m_ready(m_axi_wready),
        .m_data({output_owner,m_axi_wlast,m_axi_wstrb,m_axi_wdata}), .empty(fifo_empty)
    );

    always @(posedge clk) begin
        if(!rst_n) begin
            head<=0; tail<=0; aw_head<=0; w_head<=0;
            used<=0; aw_count<=0; w_count<=0;
            allowed<=0; aw_done<=0; w_done<=0;
            reply_valid<=0; s_axi_bid<=0; s_axi_bresp<=0;
        end else begin
            if(push) begin
                address[tail]<=s_axi_awaddr[MEMORY_ADDR_WIDTH-1:0];
                length[tail]<=s_axi_awlen; remaining[tail]<={1'b0,s_axi_awlen}+9'd1;
                identity[tail]<=s_axi_awid; cache[tail]<=s_axi_awcache; protection[tail]<=s_axi_awprot;
                allowed[tail]<=permitted(s_axi_awaddr,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,s_axi_awid);
                aw_done[tail]<=0; w_done[tail]<=0; tail<=next_ptr(tail);
            end
            if(aw_step) begin aw_done[aw_head]<=1; aw_head<=next_ptr(aw_head); end
            if(input_w) begin
                remaining[w_head]<=remaining[w_head]-1'b1;
                if(input_last) begin
                    if(!allowed[w_head]) w_done[w_head]<=1;
                    w_head<=next_ptr(w_head);
                end
            end
            if(m_axi_wvalid && m_axi_wready && m_axi_wlast) w_done[output_owner]<=1;
            if(used!=0 && aw_done[head] && w_done[head] && !reply_valid) begin
                if(!allowed[head]) begin
                    reply_valid<=1; s_axi_bid<=identity[head]; s_axi_bresp<=2'b11;
                end else if(m_axi_bvalid && m_axi_bready) begin
                    reply_valid<=1; s_axi_bid<=m_axi_bid; s_axi_bresp<=m_axi_bresp;
                end
            end
            if(retire) begin reply_valid<=0; head<=next_ptr(head); end
            case({push,retire})
                2'b10: used<=used+1'b1;
                2'b01: used<=used-1'b1;
                default: ;
            endcase
            case({push,aw_step})
                2'b10: aw_count<=aw_count+1'b1;
                2'b01: aw_count<=aw_count-1'b1;
                default: ;
            endcase
            case({push,input_last})
                2'b10: w_count<=w_count+1'b1;
                2'b01: w_count<=w_count-1'b1;
                default: ;
            endcase
        end
    end
endmodule
