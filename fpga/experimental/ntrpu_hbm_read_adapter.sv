`timescale 1ns/1ps
// AMD width/protocol conversion plus native AXI3 response checks before merging.
// A registered AR slot preserves an offered address even after a protocol fault.
module ntrpu_hbm_read_adapter (
    input wire clk,rst_n,
    input wire s_arvalid,
    output wire s_arready,
    input wire [33:0] s_araddr,
    input wire [7:0] s_arlen,
    output wire s_rvalid,
    input wire s_rready,
    output wire [511:0] s_rdata,
    output wire [3:0] s_rid,
    output wire [1:0] s_rresp,
    output wire s_rlast,
    output wire h_arvalid,
    input wire h_arready,
    output reg [33:0] h_araddr,
    output reg [3:0] h_arlen,
    output wire [5:0] h_arid,
    output wire [2:0] h_arsize,
    output wire [1:0] h_arburst,
    input wire h_rvalid,
    output wire h_rready,
    input wire [255:0] h_rdata,
    input wire [5:0] h_rid,
    input wire [1:0] h_rresp,
    input wire h_rlast,
    output wire pending,
    output reg fault
);
    wire d_arvalid,d_arready,d_rvalid,d_rready,d_rlast,d_arlock;
    wire [33:0] d_araddr;
    wire [7:0] d_arlen;
    wire [2:0] d_arsize,d_arprot;
    wire [1:0] d_arburst,d_rresp;
    wire [3:0] d_arcache,d_arqos,d_arregion;
    wire [255:0] d_rdata;
    wire p_arvalid,p_arready,p_rvalid,p_rready,p_rlast;
    wire [33:0] p_araddr;
    wire [3:0] p_arlen,p_arid;
    wire [2:0] p_arsize;
    wire [1:0] p_arburst;
    reg busy,ar_sent;
    reg [4:0] left;
    wire ar_fire=h_arvalid&&h_arready;
    wire q_valid,q_ready,q_space,q_empty;
    wire [255:0] q_data;
    wire [5:0] q_rid;
    wire [1:0] q_resp;
    wire q_last;
    // Full response metadata crosses a two-entry registered boundary before
    // classification. A malformed captured beat remains in quarantine.
    wire capture_enable=busy&&ar_sent&&!fault;
    ntrpu_hbm_read_response_queue #(.WIDTH(265)) response_cut (
        .clk(clk),.rst_n(rst_n),.s_valid(h_rvalid&&capture_enable),.s_ready(q_space),
        .s_data({h_rid,h_rresp,h_rlast,h_rdata}),.m_valid(q_valid),.m_ready(q_ready),
        .m_data({q_rid,q_resp,q_last,q_data}),.empty(q_empty));
    wire unexpected_offer=h_rvalid&&(!busy||!(ar_sent||ar_fire));
    wire bad_response=q_valid && (!busy || !ar_sent || q_rid!=0 ||
                                  q_resp==2'b01 || left==0 || q_last!=(left==1));
    wire validated_space,validated_empty;
    wire [255:0] validated_data;
    wire [5:0] validated_rid;
    wire [1:0] validated_resp;
    wire validated_last;
    // Classification precedes a second registered queue. Once offered to the
    // converter, a validated response stays valid even if a later beat faults.
    // This also breaks the VALID -> converter READY -> left-counter path.
    ntrpu_hbm_read_response_queue #(.WIDTH(265)) validated_cut (
        .clk(clk),.rst_n(rst_n),.s_valid(q_valid&&busy&&ar_sent&&!fault&&!bad_response),
        .s_ready(validated_space),.s_data({q_rid,q_resp,q_last,q_data}),
        .m_valid(p_rvalid),.m_ready(p_rready),
        .m_data({validated_rid,validated_resp,validated_last,validated_data}),.empty(validated_empty));
    assign pending=busy||!q_empty||!validated_empty;
    assign p_arready=!busy&&!fault&&q_empty&&validated_empty;
    assign h_arvalid=busy&&!ar_sent;
    assign h_arid=0;
    assign h_arsize=5;
    assign h_arburst=1;
    assign q_ready=validated_space&&busy&&ar_sent&&!fault&&!bad_response;
    assign h_rready=q_space&&capture_enable;
    always @(posedge clk)begin
        if(!rst_n)begin busy<=0;ar_sent<=0;left<=0;fault<=0;h_araddr<=0;h_arlen<=0;end
        else begin
            if(bad_response||unexpected_offer)fault<=1;
            if(p_arvalid&&p_arready)begin
                busy<=1;ar_sent<=0;left<={1'b0,p_arlen}+5'd1;h_araddr<=p_araddr;h_arlen<=p_arlen;
                if(p_arid!=0||p_arsize!=5||p_arburst!=1)fault<=1;
            end
            if(ar_fire)ar_sent<=1;
            if(q_valid&&q_ready)begin
                left<=left-1'b1;
                if(left==1)begin busy<=0;ar_sent<=0;end
            end
        end
    end
    hbm_read_width width_converter (
        .s_axi_aclk(clk),.s_axi_aresetn(rst_n),
        .s_axi_araddr(s_araddr),.s_axi_arlen(s_arlen),.s_axi_arsize(3'd6),.s_axi_arprot(3'b0),
        .s_axi_arburst(2'b01),.s_axi_arid(4'b0),.s_axi_arcache(4'b0010),.s_axi_arlock(1'b0),
        .s_axi_arregion(4'b0),.s_axi_arqos(4'b0),.s_axi_arvalid(s_arvalid),.s_axi_arready(s_arready),
        .s_axi_rdata(s_rdata),.s_axi_rid(s_rid),.s_axi_rresp(s_rresp),.s_axi_rlast(s_rlast),.s_axi_rvalid(s_rvalid),.s_axi_rready(s_rready),
        .m_axi_araddr(d_araddr),.m_axi_arlen(d_arlen),.m_axi_arsize(d_arsize),.m_axi_arprot(d_arprot),
        .m_axi_arburst(d_arburst),.m_axi_arcache(d_arcache),.m_axi_arlock(d_arlock),
        .m_axi_arregion(d_arregion),.m_axi_arqos(d_arqos),.m_axi_arvalid(d_arvalid),.m_axi_arready(d_arready),
        .m_axi_rdata(d_rdata),.m_axi_rresp(d_rresp),.m_axi_rlast(d_rlast),.m_axi_rvalid(d_rvalid),.m_axi_rready(d_rready));
    hbm_read_protocol protocol_converter (
        .aclk(clk),.aresetn(rst_n),
        .s_axi_araddr(d_araddr),.s_axi_arlen(d_arlen),.s_axi_arsize(d_arsize),.s_axi_arprot(d_arprot),
        .s_axi_arburst(d_arburst),.s_axi_arid(4'b0),.s_axi_arcache(d_arcache),.s_axi_arlock(d_arlock),
        .s_axi_arregion(d_arregion),.s_axi_arqos(d_arqos),.s_axi_arvalid(d_arvalid),.s_axi_arready(d_arready),
        .s_axi_rid(),.s_axi_rdata(d_rdata),.s_axi_rresp(d_rresp),.s_axi_rlast(d_rlast),.s_axi_rvalid(d_rvalid),.s_axi_rready(d_rready),
        .m_axi_araddr(p_araddr),.m_axi_arlen(p_arlen),.m_axi_arsize(p_arsize),.m_axi_arprot(),
        .m_axi_arburst(p_arburst),.m_axi_arid(p_arid),.m_axi_arcache(),.m_axi_arlock(),.m_axi_arqos(),
        .m_axi_arvalid(p_arvalid),.m_axi_arready(p_arready),
        .m_axi_rid(validated_rid[3:0]),.m_axi_rdata(validated_data),.m_axi_rresp(validated_resp),.m_axi_rlast(validated_last),.m_axi_rvalid(p_rvalid),.m_axi_rready(p_rready));
endmodule

// Two registered slots with independent read/write pointers. Payload write
// enables depend only on upstream admission, never on the long downstream
// RREADY path through the AMD converters. Full still inserts one admission
// bubble when popped, matching the original two-entry elastic contract.
module ntrpu_hbm_read_response_queue #(parameter integer WIDTH=265) (
    input wire clk,rst_n,
    input wire s_valid,
    output wire s_ready,
    input wire [WIDTH-1:0] s_data,
    output wire m_valid,
    input wire m_ready,
    output wire [WIDTH-1:0] m_data,
    output wire empty
);
    reg [1:0] count;
    reg write_slot,read_slot;
    reg [WIDTH-1:0] slot0,slot1;
    wire push=s_valid&&s_ready;
    wire pop=m_valid&&m_ready;
    assign s_ready=rst_n&&count<2;
    assign m_valid=rst_n&&count!=0;
    assign m_data=read_slot?slot1:slot0;
    assign empty=count==0;
    always @(posedge clk)begin
        if(!rst_n)begin count<=0;write_slot<=0;read_slot<=0;end
        else begin
            case({push,pop})
                2'b10:count<=count+1'b1;
                2'b01:count<=count-1'b1;
                default:begin end
            endcase
            if(push)begin
                if(write_slot)slot1<=s_data;
                else slot0<=s_data;
                write_slot<=!write_slot;
            end
            if(pop)read_slot<=!read_slot;
        end
    end
endmodule
